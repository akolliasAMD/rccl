/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <unistd.h>
#include "TestBed.hpp"
#include <rccl.h>

#define PIPE_WRITE(childId, val)                                        \
  ASSERT_EQ(write(childList[childId]->parentWriteFd, &val, sizeof(val)), sizeof(val))

#define PIPE_CHECK(childId)                                             \
  {                                                                     \
    int response = 0;                                                   \
    ASSERT_EQ(read(childList[childId]->parentReadFd, &response, sizeof(int)), sizeof(int)); \
    ASSERT_EQ(response, TEST_SUCCESS);                                  \
  }

namespace RcclUnitTesting
{
  TestBed::TestBed() :
    numDevicesAvailable(0),
    numActiveChildren(0),
    numActiveRanks(0)
  {
    // Set NCCL_COMM_ID to use a local port to avoid passing ncclCommId
    // Calling ncclGetUniqueId would initialize HIP, which should not be done prior to fork
    std::string localPort = "55513";
    if (!getenv("NCCL_COMM_ID"))
    {
      char hostname[HOST_NAME_MAX+1];
      gethostname(hostname, HOST_NAME_MAX+1);
      std::string hostnameString(hostname);
      hostnameString.append(":55513");
      setenv("NCCL_COMM_ID", hostnameString.c_str(), 0);
      if (ev.verbose) INFO("NCCL_COMM_ID set to %s\n", hostnameString.c_str());
    }

    // Collect the number of GPUs
    this->numDevicesAvailable = ev.maxGpus;
    if (ev.verbose) INFO("Detected %d GPUs\n", this->numDevicesAvailable);

    // Create the maximum number of possible child processes (1 per GPU)
    // Parent and child communicate via pipes
    childList.resize(this->numDevicesAvailable);
    for (int childId = 0; childId < this->numDevicesAvailable; ++childId)
    {
      childList[childId] = new TestBedChild(childId, ev.verbose, ev.printValues);
      if (childList[childId]->InitPipes() != TEST_SUCCESS)
      {
        ERROR("Unable to create pipes to child process\n");
        return;
      }

      pid_t pid = fork();
      if (pid == 0)
      {
        // Child process enters execution loop
        childList[childId]->StartExecutionLoop();
        return;
      }
      else
      {
        // Parent records child process ID and closes unused ends of pipe
        childList[childId]->pid = pid;
        close(childList[childId]->childWriteFd);
        close(childList[childId]->childReadFd);
      }
    }
  }

  void TestBed::InitComms(std::vector<std::vector<int>> const& deviceIdsPerProcess,
                          int const numCollectivesInGroup)
  {
    // Count up the total number of GPUs to use and track child/deviceId per rank
    this->numActiveChildren = deviceIdsPerProcess.size();
    this->numActiveRanks = 0;
    this->numCollectivesInGroup = numCollectivesInGroup;
    this->rankToChildMap.clear();
    this->rankToDeviceMap.clear();
    if (ev.verbose) INFO("Setting up %d active child processes\n", this->numActiveChildren);
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      for (auto i = 0; i < deviceIdsPerProcess[childId].size(); ++i)
      {
        this->rankToChildMap.push_back(childId);
        this->rankToDeviceMap.push_back(deviceIdsPerProcess[childId][i]);
        ++this->numActiveRanks;
      }
    }

    // Send InitComms command to each active child process
    int const cmd = TestBedChild::CHILD_INIT_COMMS;
    int rankOffset = 0;
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      PIPE_WRITE(childId, cmd);

      // Send total number of ranks to child process
      PIPE_WRITE(childId, this->numActiveRanks);

      // Send the rank offset for this child process
      PIPE_WRITE(childId, rankOffset);

      // Send the number of collectives to be run per group call
      PIPE_WRITE(childId, numCollectivesInGroup);

      // Send the GPUs this child uses
      int const numGpus = deviceIdsPerProcess[childId].size();
      PIPE_WRITE(childId, numGpus);
      for (int i = 0; i < numGpus; i++)
        PIPE_WRITE(childId, deviceIdsPerProcess[childId][i]);

      rankOffset += numGpus;
    }

    // Wait for child acknowledgement
    // This is done after previous loop to avoid deadlock as every rank needs to enter ncclInitCommRank
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      PIPE_CHECK(childId);
    }
  }

  void TestBed::InitComms(int const numGpus, int const numCollectivesInGroup)
  {
    InitComms(TestBed::GetDeviceIdsList(1, numGpus), numCollectivesInGroup);
  }

  void TestBed::SetCollectiveArgs(ncclFunc_t     const funcType,
                                  ncclDataType_t const dataType,
                                  ncclRedOp_t    const redOp,
                                  int            const root,
                                  size_t         const numInputElements,
                                  size_t         const numOutputElements,
                                  int            const collId,
                                  int            const rank,
                                  PtrUnion       const scalarsPerRank,
                                  int            const scalarMode)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    ScalarTransport scalarTransport;
    if (scalarMode >= 0)
    {
      ASSERT_TRUE(scalarsPerRank.ptr != NULL);

      // Capture scalars per rank in format to share with child processes
      int const numBytes = this->numActiveRanks * DataTypeToBytes(dataType);
      memcpy(scalarTransport.ptr, scalarsPerRank.ptr, numBytes);
    }

    // Loop over all ranks and send CollectiveArgs to appropriate child process
    int const cmd = TestBedChild::CHILD_SET_COLL_ARGS;
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_WRITE(childId, funcType);
      PIPE_WRITE(childId, dataType);
      PIPE_WRITE(childId, redOp);
      PIPE_WRITE(childId, root);
      PIPE_WRITE(childId, numInputElements);
      PIPE_WRITE(childId, numOutputElements);
      PIPE_WRITE(childId, scalarMode);
      PIPE_WRITE(childId, scalarTransport);
      PIPE_CHECK(childId);
    }
  }

  void TestBed::AllocateMem(bool   const inPlace,
                            bool   const useManagedMem,
                            int    const collId,
                            int    const rank)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Loop over all ranks and send allocation command to appropriate child process
    int const cmd = TestBedChild::CHILD_ALLOCATE_MEM;
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_WRITE(childId, inPlace);
      PIPE_WRITE(childId, useManagedMem);
      PIPE_CHECK(childId);
    }
  }

  void TestBed::PrepareData(int         const collId,
                            int         const rank,
                            CollFuncPtr const prepDataFunc)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    // Loop over all ranks and send prepare data command to appropriate child process
    int const cmd = TestBedChild::CHILD_PREPARE_DATA;
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_WRITE(childId, prepDataFunc);
      PIPE_CHECK(childId);
    }
  }

  void TestBed::ExecuteCollectives(std::vector<int> const &currentRanks)
  {
    int const cmd = TestBedChild::CHILD_EXECUTE_COLL;
    ++TestBed::NumTestsRun();

    std::vector<std::vector<int>> ranksPerChild(this->numActiveChildren);
    for (int rank = 0; rank < currentRanks.size(); ++rank)
    {
      ranksPerChild[rankToChildMap[currentRanks[rank]]].push_back(rank);
    }

    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      if ((currentRanks.size() == 0) || (ranksPerChild[childId].size() > 0))
      {
        PIPE_WRITE(childId, cmd);
        int tempCurrentRanks = currentRanks.size();
        PIPE_WRITE(childId, tempCurrentRanks);
        for (int rank = 0; rank < currentRanks.size(); ++rank){
          PIPE_WRITE(childId, currentRanks[rank]);
        }
      }
    }
    // Wait for child acknowledgement
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      if ((currentRanks.size() == 0) || (ranksPerChild[childId].size() > 0)) PIPE_CHECK(childId);
    }
  }

  void TestBed::ValidateResults(bool& isCorrect, int const collId, int const rank)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    int const cmd = TestBedChild::CHILD_VALIDATE_RESULTS;

    isCorrect = true;
    // Send ValidateResults command to each active child process
    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);

      int response = 0;
      ASSERT_EQ(read(childList[childId]->parentReadFd, &response, sizeof(int)), sizeof(int));
      isCorrect &= (response == TEST_SUCCESS);
    }

    ASSERT_EQ(isCorrect, true) << "Output does not match expected";
  }

  void TestBed::DeallocateMem(int const collId, int const rank)
  {
    // Build list of ranks this applies to (-1 for rank means to set for all)
    std::vector<int> rankList;
    for (int i = 0; i < this->numActiveRanks; ++i)
      if (rank == -1 || rank == i) rankList.push_back(i);

    int const cmd = TestBedChild::CHILD_DEALLOCATE_MEM;

    for (auto currRank : rankList)
    {
      int const childId = rankToChildMap[currRank];
      PIPE_WRITE(childId, cmd);
      PIPE_WRITE(childId, currRank);
      PIPE_WRITE(childId, collId);
      PIPE_CHECK(childId);
    }
  }

  void TestBed::DestroyComms()
  {
    int const cmd = TestBedChild::CHILD_DESTROY_COMMS;
    for (int childId = 0; childId < this->numActiveChildren; ++childId)
    {
      // Send DestroyComms command to each active child process
      PIPE_WRITE(childId, cmd);

      // Wait for child acknowledgement
      PIPE_CHECK(childId);
    }

    // Reset bookkeeping
    this->numActiveChildren = 0;
    this->numActiveRanks = 0;
    this->numCollectivesInGroup = 0;
  }

  void TestBed::Finalize()
  {
    // Send Stop to all child processes
    int const cmd = TestBedChild::CHILD_STOP;
    for (int childId = 0; childId < this->numDevicesAvailable; ++childId)
    {
      PIPE_WRITE(childId, cmd);

      // Close pipes to child process
      close(childList[childId]->parentWriteFd);
      close(childList[childId]->parentReadFd);
    }
    this->numDevicesAvailable = 0;
  }

  TestBed::~TestBed()
  {
    Finalize();
  }

  std::vector<ncclRedOp_t> const& TestBed::GetAllSupportedRedOps()
  {
    return ev.GetAllSupportedRedOps();
  }

  std::vector<ncclDataType_t> const& TestBed::GetAllSupportedDataTypes()
  {
    return ev.GetAllSupportedDataTypes();
  }

  std::vector<std::vector<int>> TestBed::GetDeviceIdsList(int const numProcesses,
                                                 int const numGpus)
  {
    std::vector<std::vector<int>> result(numProcesses);
    for (int i = 0; i < numGpus; i++)
      result[i % numProcesses].push_back(i);
    return result;
  }

  std::string TestBed::GetTestCaseName(int            const totalRanks,
                                       bool           const isMultiProcess,
                                       ncclFunc_t     const funcType,
                                       ncclDataType_t const dataType,
                                       ncclRedOp_t    const redOp,
                                       int            const root,
                                       bool           const inPlace,
                                       bool           const managedMem)
  {
    std::stringstream ss;
    ss << (isMultiProcess ? "MP" : "SP") <<  " ";
    ss << totalRanks << " ranks ";
    ss << ncclFuncNames[funcType] << " ";
    ss << "(" << (inPlace ? "IP" : "OP") << "," << (managedMem ? "MM" : "GM") << ") ";
    ss << ncclDataTypeNames[dataType] << " ";
    if (CollectiveArgs::UsesReduce(funcType)) ss << ncclRedOpNames[redOp] << " ";
    if (CollectiveArgs::UsesRoot(funcType)) ss << "Root " << root << " ";
    return ss.str();
  }

  void TestBed::RunSimpleSweep(std::vector<ncclFunc_t>     const& funcTypes,
                               std::vector<ncclDataType_t> const& tmpDataTypes,
                               std::vector<ncclRedOp_t>    const& tmpRedOps,
                               std::vector<int>            const& roots,
                               std::vector<int>            const& numElements,
                               std::vector<bool>           const& inPlaceList,
                               std::vector<bool>           const& managedMemList)
  {
    // Sort numElements in descending order to cut down on # of allocations
    std::vector<int> sortedN = numElements;
    std::sort(sortedN.rbegin(), sortedN.rend());

    // Filter out any unsupported datatypes, in case only subset has been compiled for
    std::vector<ncclDataType_t> const& supportedDataTypes = this->GetAllSupportedDataTypes();
    std::vector<ncclDataType_t> dataTypes;
    for (auto dt : tmpDataTypes)
    {
      for (int i = 0; i < supportedDataTypes.size(); ++i)
      {
        if (supportedDataTypes[i] == dt)
        {
          dataTypes.push_back(dt);
          break;
        }
      }
    }

    // Filter out any unsupported reduction ops, in case only subset has been compiled for
    std::vector<ncclRedOp_t> const& supportedOps = this->GetAllSupportedRedOps();
    std::vector<ncclRedOp_t> redOps;
    for (auto redop : tmpRedOps)
    {
      for (int i = 0; i < supportedOps.size(); ++i)
      {
        if (supportedOps[i] == redop)
        {
          redOps.push_back(redop);
          break;
        }
      }
    }

    bool isCorrect = true;

    // Sweep over the number of ranks
    for (int totalRanks = ev.minGpus; totalRanks <= ev.maxGpus && isCorrect; ++totalRanks)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(ev.processMask & (1 << isMultiProcess))) continue;

      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      this->InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks));

      for (int ftIdx = 0; ftIdx < funcTypes.size()      && isCorrect; ++ftIdx)
      for (int dtIdx = 0; dtIdx < dataTypes.size()      && isCorrect; ++dtIdx)
      for (int rdIdx = 0; rdIdx < redOps.size()         && isCorrect; ++rdIdx)
      for (int rtIdx = 0; rtIdx < roots.size()          && isCorrect; ++rtIdx)
      for (int ipIdx = 0; ipIdx < inPlaceList.size()    && isCorrect; ++ipIdx)
      for (int mmIdx = 0; mmIdx < managedMemList.size() && isCorrect; ++mmIdx)
      {
        if (ev.showNames)
        {
          std::string name = this->GetTestCaseName(totalRanks, isMultiProcess,
                                                   funcTypes[ftIdx], dataTypes[dtIdx],
                                                   redOps[rdIdx], roots[rtIdx],
                                                   inPlaceList[ipIdx], managedMemList[mmIdx]);
          INFO("%s\n", name.c_str());
        }

        for (int neIdx = 0; neIdx < numElements.size() && isCorrect; ++neIdx)
        {
          int numInputElements, numOutputElements;
          CollectiveArgs::GetNumElementsForFuncType(funcTypes[ftIdx],
                                                    sortedN[neIdx],
                                                    totalRanks,
                                                    &numInputElements,
                                                    &numOutputElements);

          this->SetCollectiveArgs(funcTypes[ftIdx],
                                  dataTypes[dtIdx],
                                  redOps[rdIdx],
                                  roots[rtIdx],
                                  numInputElements,
                                  numOutputElements);

          // Only allocate once for largest size
          if (neIdx == 0) this->AllocateMem(inPlaceList[ipIdx], managedMemList[mmIdx]);

          // There are some cases when data does not need to be re-prepared
          // e.g. AllReduce subarray expected results are still valid
          bool canSkip = (neIdx != 0 && !inPlaceList[ipIdx] &&
                          (funcTypes[ftIdx] == ncclCollBroadcast ||
                           funcTypes[ftIdx] == ncclCollReduce    ||
                           funcTypes[ftIdx] == ncclCollAllReduce));
          if (!canSkip) this->PrepareData();

          this->ExecuteCollectives();
          this->ValidateResults(isCorrect);
          if (!isCorrect)
          {
            std::string name = this->GetTestCaseName(totalRanks, isMultiProcess,
                                                     funcTypes[ftIdx], dataTypes[dtIdx],
                                                     redOps[rdIdx], roots[rtIdx],
                                                     inPlaceList[ipIdx], managedMemList[mmIdx]);
            ERROR("Incorrect output for %s\n", name.c_str());
          }
        }
        this->DeallocateMem();
      }
      this->DestroyComms();
    }
  }

  int& TestBed::NumTestsRun()
  {
    static int numTestsRun = 0;
    return numTestsRun;
  }
}

#undef PIPE_WRITE
#undef PIPE_CHECK
