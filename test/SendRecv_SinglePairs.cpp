/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(SendRecv, SinglePairs)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;

    int numCollPerGroup = 0;
    bool isCorrect = true;
    int totalRanks = testBed.ev.maxGpus;
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess) // CHANGE TO SINGLE
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks), 1);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      {
        for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
        {
          // Run all element sizes in parallel as single group
          for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
          {
            for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
            {
              testBed.SetCollectiveArgs(ncclCollSend,
                                        dataTypes[dataIdx],
                                        ncclSum, // This should be moved to optional variables struct
                                        recvRank,
                                        numElements[numIdx],
                                        numElements[numIdx],
                                        0,
                                        sendRank);
              if (recvRank == 0)
              {

                testBed.AllocateMem(inPlace, useManagedMem, -1, sendRank);
                testBed.PrepareData(-1, sendRank);
              }
              if (recvRank  != sendRank)
              {
                if (testBed.ev.showNames) // Show test names
                  INFO("%s process Datatype: %s SendReceive test Rank %d -> Rank %d for %d number of Elements\n",
                      isMultiProcess ? "Multi " : "Single",
                      ncclDataTypeNames[dataTypes[dataIdx]],
                      sendRank,
                      recvRank,
                      numElements[numIdx]);


                testBed.SetCollectiveArgs(ncclCollRecv,
                                          dataTypes[dataIdx],
                                          ncclSum, // This should be moved to optional variables struct
                                          sendRank,
                                          numElements[numIdx],
                                          numElements[numIdx],
                                          0,
                                          recvRank);
                testBed.AllocateMem(inPlace, useManagedMem, -1, recvRank);
                testBed.PrepareData(-1, recvRank);
                testBed.ExecuteCollectives({sendRank,recvRank });
                testBed.ValidateResults(isCorrect, -1, recvRank);
                testBed.DeallocateMem(-1, recvRank);
              }

            }
            testBed.DeallocateMem(-1, sendRank);
          }
        }
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }
}
