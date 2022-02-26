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
    std::vector<ncclFunc_t>     const  funcType        = {ncclCollSend, ncclCollRecv}; // akollias
    // simple send receive test would check that communication is going for rank a to rank b
    // you need to do send and receive without caring about redOps, but care about just data...
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32};
    std::vector<ncclRedOp_t>    const& redOps          = {ncclSum}; //Not important for send receive tests
    std::vector<int>            const  numElements     = {1024}; // akollias, one test for now
    // std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    // int                         const  root            = 0;
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    int                         const  numCollPerGroup = numElements.size();

    // CollFuncPtr prepFunc = DefaultPrepData_SendRecv; // akollias
    bool isCorrect = true;
    // for (int totalRanks = testBed.ev.minGpus; totalRanks <= testBed.ev.maxGpus && isCorrect; ++totalRanks) // big iterator for all ranks
    int totalRanks = testBed.ev.maxGpus; // akollias this to change on maxGpus
    // int totalRanks = 2; // akollias this to change on maxGpus
    // for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess) // akollias disable multi process in the beggining enable after
    // { // akollias multiprocess
      // Test either single process all GPUs, or 1 process per GPU
      // int const numProcesses = isMultiProcess ? totalRanks : 1; // one process for the moment
      int const numProcesses = 1;
      int const isMultiProcess = 0;
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
        // for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx) // akollias for now we are removing differnt collectives
        for (int root = 0; root < totalRanks; ++root)
        {
          testBed.SetCollectiveArgs(funcType[0],
                                    dataTypes[dataIdx],
                                    redOps[0],
                                    0, // 0?
                                    numElements[0],
                                    numElements[0],
                                    0,
                                    root);
          testBed.AllocateMem(inPlace, useManagedMem, -1, root); // need to have rank
          testBed.PrepareData(-1, root); // need to have rank
          for (int currentRank = 0; currentRank < totalRanks; ++currentRank)
          { // so root needs the recv number, and recv gpu needs the root rank

            if (currentRank != root)
            {
              testBed.SetCollectiveArgs(funcType[0],
                                        dataTypes[dataIdx],
                                        redOps[0],
                                        currentRank,
                                        numElements[0],
                                        numElements[0],
                                        0,
                                        root);

              testBed.SetCollectiveArgs(funcType[1],
                                        dataTypes[dataIdx],
                                        redOps[0],
                                        root,
                                        numElements[0],
                                        numElements[0],
                                        0,
                                        currentRank);
              testBed.AllocateMem(inPlace, useManagedMem, -1, currentRank); // need to have currentRank
              testBed.PrepareData(-1, currentRank); // need to have currentRank
              testBed.ExecuteCollectives({root,currentRank});// specifically for all of them // need to have currentRank
              // testBed.ExecuteCollectives();// specifically for all of them // need to have currentRank
              testBed.ValidateResults(isCorrect, -1, currentRank); // need to have currentRank
              testBed.DeallocateMem(-1, currentRank); // need to have currentRank
              // testBed.DestroyComms();
            }

          }
          // testBed.ValidateResults(isCorrect, -1, root); // need to have root
          testBed.DeallocateMem(-1, root); // need to have root
        }
      }
      testBed.DestroyComms();
    // } // akollias multiprocess
    testBed.Finalize();
  }
}
