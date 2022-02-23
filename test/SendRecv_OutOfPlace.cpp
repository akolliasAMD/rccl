/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(SendRecv, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    // ncclFunc_t                  const  funcType        = ncclCollAllReduce;
    ncclFunc_t                  const  funcType        = ncclCollSendRecv; // akollias
    // simple send receive test would check that communication is going for rank a to rank b
    // you need to do send and receive without caring about redOps, but care about just data...
    std::vector<ncclDataType_t> const& dataTypes       = {ncclFloat};
    // std::vector<ncclRedOp_t>    const& redOps          = {ncclSum}; //Not important for send receive tests
    std::vector<int>            const  numElements     = {1024}; // akollias, one test for now
    // std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    int                         const  root            = 0;
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    int                         const  numCollPerGroup = numElements.size();

    // This tests runs 3 collectives in the same group call


    // akollias

    // need to be in group calls 
          // ncclSend     (const void* sendbuff,                 size_t count, ncclDataType_t datatype, int peer,       ncclComm_t comm, hipStream_t stream);
          // ncclRecv     (      void* recvbuff,                 size_t count, ncclDataType_t datatype, int peer,       ncclComm_t comm, hipStream_t stream);
          // ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream);
    // initComms for 2 devices (later for all)
    // testBed.SetCollectiveArgs(); // send receive 
    // testBed.AllocateMem(inPlace, useManagedMem);
    // testBed.PrepareData();
    // testBed.ExecuteCollectives();
    // testBed.ValidateResults(isCorrect);
    // testBed.DeallocateMem();
    // 

    // CollFuncPtr prepFunc = DefaultPrepData_SendRecv; // akollias
    bool isCorrect = true;
    // for (int totalRanks = testBed.ev.minGpus; totalRanks <= testBed.ev.maxGpus && isCorrect; ++totalRanks) // big iterator for all ranks
    int totalRanks = 2 // akollias this to change on maxGpus
    // for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess) // akollias disable multi process in the beggining enable after
    // { // akollias multiprocess
      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks), numCollPerGroup);

      // for (int redOpIdx = 0; redOpIdx < redOps.size() && isCorrect; ++redOpIdx)
      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      {
        if (testBed.ev.showNames) // Show test names
          INFO("%s process %2d-ranks SendRec v%d out Of place (%s)\n",
               isMultiProcess ? "Multi " : "Single",
               numCollPerGroup,
               totalRanks, ncclDataTypeNames[dataTypes[dataIdx]]); // akollias no Red_ops

        // Run all element sizes in parallel as single group // akollias for now this will only run for one test
        for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
        {
          testBed.SetCollectiveArgs(funcType, //some of them should be send some should be rec?
                                    dataTypes[dataIdx],
                                    // redOps[redOpIdx], // akollias this does not exist but has no default value
                                    -1, // akollias instead of redops ??
                                    root, // akollias this will probably need to change to takje into  consideration the other ones needing to send
                                    numElements[collIdx], // akollias here for first time it will all be 1024
                                    numElements[collIdx],
                                    collIdx);
        }
        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData(); // Default prep RecSend
        testBed.ExecuteCollectives();
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();
      }
      testBed.DestroyComms();
    // } // akollias multiprocess
    testBed.Finalize();
  }
}
