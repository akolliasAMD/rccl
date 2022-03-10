/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(AllToAllv, OutOfPlace)
  {
    TestBed testBed;


    // Configuration
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;


    OptionalColArgs allToAllvCounts;
    int numCollPerGroup = 0;
    bool isCorrect = true;
    int totalRanks = testBed.ev.maxGpus;

    // send and receive prep, should be put in a function
    for (int root = 0; root < totalRanks; ++root)
    {
      for (int curRank = 0; curRank  < totalRanks; ++curRank)
      {
        //create send counts, and build other arrays from that
        allToAllvCounts.sendcounts[root*totalRanks + curRank] = numElements[0] * (curRank + 1);
        if (root == curRank) allToAllvCounts.sendcounts[root*totalRanks + curRank] = 0;
        allToAllvCounts.recvcounts[curRank*totalRanks + root] = allToAllvCounts.sendcounts[root*totalRanks + curRank];
      }
    }
    for (int root = 0; root < totalRanks; ++root)
    {
      allToAllvCounts.sdispls[root*totalRanks] = 0;
      allToAllvCounts.rdispls[root*totalRanks] = 0;
      for (int curRank = 1; curRank  < totalRanks; ++curRank)
      {
        allToAllvCounts.sdispls[root*totalRanks + curRank] = allToAllvCounts.sdispls[root*totalRanks + curRank - 1] + allToAllvCounts.sendcounts[root*totalRanks + curRank - 1];
        allToAllvCounts.rdispls[root*totalRanks + curRank] = allToAllvCounts.rdispls[root*totalRanks + curRank - 1] + allToAllvCounts.recvcounts[root*totalRanks + curRank - 1];
      }
      allToAllvCounts.numInputElementsArray[root] = allToAllvCounts.sdispls[(root+ 1)*totalRanks - 1] + allToAllvCounts.sendcounts[(root+ 1)*totalRanks - 1];
      allToAllvCounts.numOutputElementsArray[root] = allToAllvCounts.rdispls[(root+ 1)*totalRanks - 1] + allToAllvCounts.recvcounts[(root+ 1)*totalRanks - 1];
    }


    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks));

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      {
        if (testBed.ev.showNames)
        {
          std::string name = testBed.GetTestCaseName(totalRanks, isMultiProcess,
                                                     ncclCollAllToAllv, dataTypes[dataIdx],
                                                     ncclSum, -1,
                                                     inPlace, useManagedMem);
          INFO("%s\n", name.c_str());

        }
        testBed.SetCollectiveArgs(ncclCollAllToAllv,
                                  dataTypes[dataIdx],
                                  ncclSum, // This should be moved to optional variables struct
                                  0, //does not affect anything
                                  0,
                                  0,
                                  allToAllvCounts);
        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData(); // fails in here
        testBed.ExecuteCollectives();
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();

      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }
}
