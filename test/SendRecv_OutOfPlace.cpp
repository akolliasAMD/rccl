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
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;


    int numCollPerGroup = 0;
    bool isCorrect = true;
    int totalRanks = testBed.ev.maxGpus;
    int const numProcesses = 1;
    int const isMultiProcess = 0; //multi process is the next step
    testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks), 0);

    for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
    {
      // Run all element sizes in parallel as single group
      for (int root = 0; root < totalRanks; ++root)
      {
        testBed.SetCollectiveArgs(ncclCollSend,
                                  dataTypes[dataIdx],
                                  ncclSum, // This should be moved to optional variables struct
                                  0, // 0?
                                  numElements[0],
                                  numElements[0],
                                  0,
                                  root);
        testBed.AllocateMem(inPlace, useManagedMem, -1, root);
        testBed.PrepareData(-1, root);
        for (int currentRank = 0; currentRank < totalRanks; ++currentRank)
        {

          if (currentRank != root)
          {
            if (testBed.ev.showNames) // Show test names
              INFO("%s process SendReceive test Rank %d -> Rank %d\n",
                  isMultiProcess ? "Multi " : "Single",
                  root,
                  currentRank);
            testBed.SetCollectiveArgs(ncclCollSend,
                                      dataTypes[dataIdx],
                                      ncclSum, // This should be moved to optional variables struct
                                      currentRank,
                                      numElements[0],
                                      numElements[0],
                                      0,
                                      root);

            testBed.SetCollectiveArgs(ncclCollRecv,
                                      dataTypes[dataIdx],
                                      ncclSum, // This should be moved to optional variables struct
                                      root,
                                      numElements[0],
                                      numElements[0],
                                      0,
                                      currentRank);
            testBed.AllocateMem(inPlace, useManagedMem, -1, currentRank);
            testBed.PrepareData(-1, currentRank);
            testBed.ExecuteCollectives({root,currentRank});
            testBed.ValidateResults(isCorrect, -1, currentRank);
            testBed.DeallocateMem(-1, currentRank);
          }

        }
        testBed.DeallocateMem(-1, root);
      }
    }
    testBed.DestroyComms();
    testBed.Finalize();
  }
}