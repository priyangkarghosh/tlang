#define BLOCK_SIZE 256u
#define SUBGROUP_SIZE 32u
#define RADIX_BITS 4u
#define RADIX_BUCKETS (1u << RADIX_BITS)

static const uint UINT_MAX = 0xffffffffu;
static const uint NUM_SUBGROUPS = BLOCK_SIZE / SUBGROUP_SIZE;
static const uint SG_HIST_SIZE = NUM_SUBGROUPS * RADIX_BUCKETS;

struct RS_Resources {
    RWStructuredBuffer<uint2> input;
    RWStructuredBuffer<uint2> output;
    RWStructuredBuffer<uint> device_hist;

    uint numElements, numWorkGroups;
    uint mask, radixBits, shift;
};

// bind at resources at b0
ParameterBlock<RS_Resources> res : register(b0);
ParameterBlock<RS_Resources> res2 : register(b1);

// group memory
groupshared uint group_hist[RADIX_BUCKETS];

[shader("compute"), numthreads(BLOCK_SIZE, 1, 1)]
void buildDeviceHistogram(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID    : SV_GroupThreadID,
    uint3 groupID          : SV_GroupID,
)
{
    // shorthand aliases
    uint gid = dispatchThreadID.x;
    uint tid = groupThreadID.x;

    // each thread zeroes one (or more) buckets
#if RADIX_BUCKETS <= BLOCK_SIZE
    if (tid < RADIX_BUCKETS) group_hist[tid] = 0;
#else
    for (uint i = tid; i < RADIX_BUCKETS; i += BLOCK_SIZE) 
        group_hist[i] = 0;
#endif
    GroupMemoryBarrierWithGroupSync();

    // 2) make a tiny register-array histogram
    uint localHist[RADIX_BUCKETS];
    [unroll]
    for (uint b = 0; b < RADIX_BUCKETS; b++)
        localHist[b] = 0u;

    // 3) each thread accumulates its elements into localHist
    uint idx     = dispatchThreadID.x;
    uint stride  = BLOCK_SIZE * res.numWorkGroups;
    for (uint i = idx; i < res.numElements; i += stride)
    {
        uint key    = res.input[i].x;
        uint bucket = (key >> res.shift) & res.mask;
        localHist[bucket]++;
    }

    // 4) reduce each bucket across the current subgroup (warp)
    //    WaveActiveSum sums `localHist[b]` over all lanes.
    [unroll]
    for (uint b = 0; b < RADIX_BUCKETS; b++)
    {
        uint sum = WaveActiveSum(localHist[b]);
        // let lane 0 do exactly one atomic into shared memory
        if (WaveGetLaneIndex() == 0)
            InterlockedAdd(group_hist[b], sum);
    }
    GroupMemoryBarrierWithGroupSync();

    // write the per-group histograms into the global buffer
#if RADIX_BUCKETS <= BLOCK_SIZE
    if (tid < RADIX_BUCKETS) res.device_hist[tid * res.numWorkGroups + groupID.x] = group_hist[tid];
#else
    for (uint i = tid; i < RADIX_BUCKETS; i += BLOCK_SIZE)
        res.device_hist[i * numWorkGroups.x + groupID.x] = group_hist[i];
#endif
}