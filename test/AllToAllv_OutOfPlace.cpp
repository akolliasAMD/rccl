/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{

  void sendRecvPrep(size_t numInputElementsArray[], size_t numOutputElementsArray[], OptionalColArgs &options, int totalRanks, int numElementsBase)
  {
    for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
    {
      for (int recvRank  = 0; recvRank   < totalRanks; ++recvRank )
      {
        //create send counts, and build other arrays from that
        options.sendcounts[sendRank*totalRanks+recvRank] = numElementsBase * (recvRank + 1);
        options.recvcounts[recvRank*totalRanks+sendRank] = options.sendcounts[sendRank*totalRanks+recvRank ];
      }
    }
    for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
    {
      options.sdispls[sendRank*totalRanks] = 0;
      options.rdispls[sendRank*totalRanks] = 0;
      for (int recvRank = 1; recvRank < totalRanks; ++recvRank )
      {
        options.sdispls[sendRank*totalRanks+recvRank] = options.sdispls[sendRank*totalRanks+recvRank-1] + options.sendcounts[sendRank*totalRanks+recvRank-1];
        options.rdispls[sendRank*totalRanks+recvRank] = options.rdispls[sendRank*totalRanks+recvRank-1] + options.recvcounts[sendRank*totalRanks+recvRank-1];
      }
      numInputElementsArray[sendRank] = options.sdispls[(sendRank+1)*totalRanks-1] + options.sendcounts[(sendRank+1)*totalRanks-1];
      numOutputElementsArray[sendRank] = options.rdispls[(sendRank+1)*totalRanks-1] + options.recvcounts[(sendRank+1)*totalRanks-1];
    }

  }

  TEST(AllToAllv, OutOfPlace)
  {
    TestBed testBed;
    // Configuration
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclFloat64};
    std::vector<int>            const  numElementsBase = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;

    OptionalColArgs options;
    size_t numInputElementsArray[MAX_RANKS], numOutputElementsArray[MAX_RANKS];
    bool isCorrect = true;
    int totalRanks = testBed.ev.maxGpus;

    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks));

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElementsBase.size() && isCorrect; ++numIdx)
      {
        if (testBed.ev.showNames)
        {
          std::string name = testBed.GetTestCaseName(totalRanks, isMultiProcess,
                                                     ncclCollAllToAllv, dataTypes[dataIdx],
                                                     ncclSum, -1,
                                                     inPlace, useManagedMem);
          INFO("%s\n", name.c_str());

        }
        sendRecvPrep(numInputElementsArray, numOutputElementsArray, options, totalRanks, numElementsBase[numIdx]);
        for (int rank = 0; rank < totalRanks; ++rank)
        {
          testBed.SetCollectiveArgs(ncclCollAllToAllv,
                                  dataTypes[dataIdx],
                                  numInputElementsArray[rank],
                                  numOutputElementsArray[rank],
                                  -1,
                                  rank,
                                  options);

        }
        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData();
        testBed.ExecuteCollectives();
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();

      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }
}
