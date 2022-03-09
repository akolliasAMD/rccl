/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(AllToAllV, OutOfPlace)
  {
    TestBed testBed;


    // Configuration
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32};
    std::vector<int>            const  numElements     = {100}; //{1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;

    size_t sendcounts[MAX_RANKS * MAX_RANKS];
    size_t sdispls[MAX_RANKS * MAX_RANKS];
    size_t recvcounts[MAX_RANKS * MAX_RANKS];
    size_t rdispls[MAX_RANKS * MAX_RANKS];
    size_t numInputElements[MAX_RANKS];
    size_t numOutputElements[MAX_RANKS];

    int numCollPerGroup = 0;
    bool isCorrect = true;
    int totalRanks = testBed.ev.maxGpus;

    //data prep needs to take into consideration sdispl and r displ (mostly rdispl)
    //Allocate mem too will need send receive (maybe i can add a function to the struct, it is the sum of all things) (EVERY TIME THE COLLECTIVE IS CALLED IT IS CALLED WITH RANK MEM)

    // send and receive prep, should be put in a function
    for (int root = 0; root < totalRanks; ++root)
    {
      for (int curRank = 0; curRank  < totalRanks; ++curRank)
      {
        //create send counts, figure it out from there should change num elemends if you want more complicated

        sendcounts[root*totalRanks + curRank] = numElements[0] * (curRank + 1);
        if (root == curRank) sendcounts[root*totalRanks + curRank] = 0;
        // sendcounts[root*totalRanks + curRank] = numElements[0];
        recvcounts[curRank*totalRanks + root] = sendcounts[root*totalRanks + curRank];
      }
    }
    for (int root = 0; root < totalRanks; ++root)
    {
      sdispls[root*totalRanks] = 0;
      rdispls[root*totalRanks] = 0;
      for (int curRank = 1; curRank  < totalRanks; ++curRank)
      {
        sdispls[root*totalRanks + curRank] = sdispls[root*totalRanks + curRank - 1] + sendcounts[root*totalRanks + curRank - 1];
        rdispls[root*totalRanks + curRank] = rdispls[root*totalRanks + curRank - 1] + recvcounts[root*totalRanks + curRank - 1];
      }
      // akollias  sdispls[root*totalRanks + totalRanks - 1] + sendcounts[root*totalRanks + totalRanks - 1];
      numInputElements[root] = sdispls[(root+ 1)*totalRanks - 1] + sendcounts[(root+ 1)*totalRanks - 1];
      numOutputElements[root] = rdispls[(root+ 1)*totalRanks - 1] + recvcounts[(root+ 1)*totalRanks - 1];
    }


    for (int root = 0; root < totalRanks; ++root)
    {
      for (int curRank = 0; curRank  < totalRanks; ++curRank)
        printf("%lu, ", recvcounts[root*totalRanks + curRank]);
      printf("\n");
    }

    // for (int root = 0; root < totalRanks; ++root)
    // {
    //   for (int curRank = 0; curRank  < totalRanks; ++curRank)
    //     printf("%lu, ", rdispls[root*totalRanks + curRank]);
    //   printf("\n");
    // }
    // for (int root = 0; root < totalRanks; ++root)
    // {
    //   INFO("%lu send, %lu Receive, %d rank\n", numInputElements[root], numOutputElements[root], root);

    // }

    for (int isMultiProcess = 0; isMultiProcess <= 0 && isCorrect; ++isMultiProcess) //akollias change to one should be fine
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks), 1);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      {
        testBed.SetCollectiveArgs(ncclCollAllToAllv,
                                  dataTypes[dataIdx],
                                  ncclSum, // This should be moved to optional variables struct
                                  0, //does not affect anything
                                  0,
                                  0,
                                  -1,
                                  -1,
                                  {nullptr},
                                  -1,
                                  sendcounts, //akollias
                                  sdispls,
                                  recvcounts,
                                  rdispls,
                                  numInputElements,
                                  numOutputElements);
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
