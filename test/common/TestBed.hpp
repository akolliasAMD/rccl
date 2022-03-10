/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once
#include <map>
#include "CollectiveArgs.hpp"
#include "TestBedChild.hpp"
#include "EnvVars.hpp"
#include <gtest/gtest.h>

namespace RcclUnitTesting
{
  // This class facilitates testing RCCL collectives across various process / device configurations
  //
  class TestBed
  {
  public:
    int                        numDevicesAvailable;   // # of devices detected on node
    std::vector<TestBedChild*> childList;             // List of child processes
    std::vector<int>           rankToChildMap;        // Tracks which child process each rank is assigned to
    std::vector<int>           rankToDeviceMap;       // Tracks which device each rank is assigned to
    int                        numActiveChildren;     // List of active children (with usable RCCL comms)
    int                        numActiveRanks;        // Current # of ranks in use
    int                        numCollectivesInGroup; // # of collectives to execute per group call

    EnvVars ev;                                       // Environment variables

    // Constructor - Creates one child process per detected GPU device that waits for further commands
    TestBed();

    // Prepare TestBed for use with GPUs across multiple child processes
    void InitComms(std::vector<std::vector<int>> const& deviceIdsPerChild, int const numCollectivesInGroup = 1);
    // Prepare TestBed for use with GPUs on a single child process
    void InitComms(int const numGpus, int const numCollectivesInGroup = 1);

    // Set collectives arguments for specified collective / rank
    // Setting scalarsPerRank to non-null will create custom reduction operator
    // Using collId = -1 (default) applies settings to all collectives in group
    // Using rank = -1 (default) applies settings to all ranks

    void SetCollectiveArgs(ncclFunc_t     const funcType,
                           ncclDataType_t const dataType,
                           ncclRedOp_t    const redOp,
                           int            const root,
                           size_t         const numInputElements,
                           size_t         const numOutputElements,
                           OptionalColArgs const &optionalArgs = {});



    // Allocate memory for specified collective / rank
    // - Requires SetCollectiveArgs to have been called already
    // Using collId = -1 (default) applies settings to all collectives in group
    // Using rank = -1 (default) applies settings to all ranks
    void AllocateMem(bool   const inPlace = false,
                     bool   const useManagedMem = false,
                     int    const collId = -1,
                     int    const rank = -1);

    // Initialize input and compute expected results
    // - requires that SetCollectiveArgs and AllocateMemory have already been called
    // Setting collId to -1 applies settings to all collectives in group
    // Setting rank to -1 applies settings to all ranks
    // Setting prepDataFunc to nullptr uses the default fill pattern routine
    void PrepareData(int const collId = -1,
                     int const rank = -1,
                     CollFuncPtr const prepDataFunc = nullptr);

    // Execute all collectives on all test children
    // Blocks until collective is completed
    void ExecuteCollectives(std::vector<int> const &currentRanks = {});

    // Perform results validation - compare output to expected
    void ValidateResults(bool& isCorrect, int collId = -1, int const rank = -1);

    // Release allocated memory
    void DeallocateMem(int collId = -1, int const rank = -1);

    // Release the RCCL comms
    void DestroyComms();

    // Explicit TestBed destructor that releases all child processes
    // No further calls to TestBed should be performed after this call
    void Finalize();

    // Destructor - Calls Finalize() to release all child processes
    ~TestBed();

    // Returns all the supported reduction operations based on build settings
    std::vector<ncclRedOp_t> const& GetAllSupportedRedOps();

    // Return all the supported data types based on build settings
    std::vector<ncclDataType_t> const& GetAllSupportedDataTypes();

    // Helper function that splits up GPUs to the given number of processes
    static std::vector<std::vector<int>> GetDeviceIdsList(int const numProcesses,
                                                          int const numGpus);

    // Generate a test case name
    static std::string GetTestCaseName(int            const totalRanks,
                                       bool           const isMultiProcess,
                                       ncclFunc_t     const funcType,
                                       ncclDataType_t const dataType,
                                       ncclRedOp_t    const redOp,
                                       int            const root,
                                       bool           const inPlace,
                                       bool           const managedMem);

    // Run a simple sweep
    void RunSimpleSweep(std::vector<ncclFunc_t>     const& funcTypes,
                        std::vector<ncclDataType_t> const& dataTypes,
                        std::vector<ncclRedOp_t>    const& redOps,
                        std::vector<int>            const& roots,
                        std::vector<int>            const& numElements,
                        std::vector<bool>           const& inPlaceList,
                        std::vector<bool>           const& managedMemList);

    // Used to track total number of calls to ExecuteCollectives()
    static int& NumTestsRun();

  protected:
    // Ends the specified child process
    void StopChild(int const childId);
  };
}
