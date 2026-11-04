// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Regression test for the BatchDesc leak on the InvalidArgument submit path.
//
// Background: MultiTransport::mp_submitTransfer() grows
// batch_desc.task_list before running mp_selectTransport() for each request.
// When a request fails selection (e.g. the target segment descriptor is
// missing or stale -- a peer restarted on the same ip:port, or announced its
// segment before registering its memory regions), the function returned
// early, leaving the freshly appended tasks with is_finished == false. Those
// tasks were never dispatched and will never complete, so freeBatchID()
// refuses to delete the BatchDesc (BatchBusy) and defers it to the cleanup
// worker -- which can never reclaim it either. The result is one leaked
// BatchDesc per failed submit, accumulating for as long as the bad target is
// retried.
//
// The fix rolls back task_list to its pre-submit size when
// mp_selectTransport() fails. This test hammers a bad target id through both
// the multi-protocol entry point (when built with ENABLE_MULTI_PROTOCOL, the
// path that actually had the leak) and the regular submitTransfer() entry
// point, asserting, deterministically per iteration, that:
//   1. the submit fails with InvalidArgument (the bad-address path is hit),
//      and
//   2. freeBatchID() returns OK (the BatchDesc is reclaimed, i.e. no leak).
// Without the fix, assertion (2) fails on the first multi-protocol
// iteration.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "error.h"
#include "multi_transport.h"
#include "transfer_metadata.h"
#include "transport/transport.h"

using namespace mooncake;

namespace {

// Builds a request targeting a segment id that was never opened:
// getSegmentDescByID() returns null and *selectTransport() returns
// InvalidArgument. This is exactly the "bad address" produced by the
// peer-restart / premature-mount scenario.
Transport::TransferRequest MakeBadAddressRequest() {
    Transport::TransferRequest request;
    request.opcode = Transport::TransferRequest::READ;
    request.source = reinterpret_cast<void *>(0x1000);
    request.target_id = 999999;
    request.target_offset = 0;
    request.length = 4096;
    return request;
}

}  // namespace

TEST(MultiTransportSubmitLeakTest, InvalidArgumentDoesNotLeakBatchDesc) {
    // P2PHANDSHAKE mode: the TransferMetadata constructor returns without
    // contacting any metadata server, so this test needs no RDMA device and
    // no running master -- it exercises pure in-process bookkeeping logic.
    auto metadata = std::make_shared<TransferMetadata>(P2PHANDSHAKE);
    std::string local_server_name = "127.0.0.1:0";
    MultiTransport multi{metadata, local_server_name};

    constexpr int kIterations = 2000;
    const auto bad_request = MakeBadAddressRequest();

#ifdef ENABLE_MULTI_PROTOCOL
    // The fixed path: mp_submitTransfer() grows task_list before selecting
    // transports, so a selection failure used to leak the BatchDesc.
    std::string proto = "rdma";
    for (int i = 0; i < kIterations; ++i) {
        Transport::BatchID batch_id = multi.allocateBatchID(1);
        ASSERT_NE(batch_id, static_cast<Transport::BatchID>(ERR_MEMORY))
            << "allocateBatchID failed at iteration " << i;

        std::vector<Transport::TransferRequest> requests{bad_request};
        Status submit_status =
            multi.mp_submitTransfer(batch_id, requests, proto);
        ASSERT_TRUE(submit_status.IsInvalidArgument())
            << "iteration " << i << ": expected InvalidArgument, got code="
            << static_cast<int>(submit_status.code());

        Status free_status = multi.freeBatchID(batch_id);
        ASSERT_TRUE(free_status.ok())
            << "iteration " << i
            << ": freeBatchID did not reclaim the BatchDesc (leak), code="
            << static_cast<int>(free_status.code());
    }
#endif

    // The regular entry point must keep the same contract: a failed submit
    // never leaves the batch un-reclaimable.
    for (int i = 0; i < kIterations; ++i) {
        Transport::BatchID batch_id = multi.allocateBatchID(1);
        ASSERT_NE(batch_id, static_cast<Transport::BatchID>(ERR_MEMORY))
            << "allocateBatchID failed at iteration " << i;

        std::vector<Transport::TransferRequest> requests{bad_request};
        Status submit_status = multi.submitTransfer(batch_id, requests);
        ASSERT_TRUE(submit_status.IsInvalidArgument())
            << "iteration " << i << ": expected InvalidArgument, got code="
            << static_cast<int>(submit_status.code());

        Status free_status = multi.freeBatchID(batch_id);
        ASSERT_TRUE(free_status.ok())
            << "iteration " << i
            << ": freeBatchID did not reclaim the BatchDesc (leak), code="
            << static_cast<int>(free_status.code());
    }
}
