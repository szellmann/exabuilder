#include <bitset>
#include <fstream>
#include <iostream>
#include <vector>
#include <cub/cub.cuh>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "math.h"
#include "timer.h"

// App state
static std::string g_inFileName;
static std::string g_outFileName;
enum Mode { Morton, Projection };
static Mode g_mode{Projection};

// Currently not relevant to us yet; only once we
// decide to include cases where clusters are
// allowed to overlap:
// #define CAN_OVERLAP

// print cell and brick bounds for debugging after builder finished
#define DO_PRINT_BOUNDS     0
// exec nvidia-smi at the end to monitor mem usage
#define DO_EXEC_NVIDIA_SMI  0

#ifndef NDEBUG
#define CUDA_SAFE_CALL(FUNC) { cuda_safe_call((FUNC), __FILE__, __LINE__); }
#else
#define CUDA_SAFE_CALL(FUNC) FUNC
#endif

inline void cuda_safe_call(cudaError_t code, char const* file, int line, bool fatal = false)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s %s:%i\n", cudaGetErrorString(code), file, line);
    if (fatal)
      exit(code);
  }
}

#define CUDA_SYNC_CHECK(void) CUDA_SAFE_CALL(cudaDeviceSynchronize())

#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif

inline std::string exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
#ifdef _WIN32
  std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
#else
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
#endif
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

inline std::string prettyBytes(const size_t s)
{
  char buf[1000];
  if (s >= (1024LL*1024LL*1024LL*1024LL)) {
    osp_snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
  } else if (s >= (1024LL*1024LL*1024LL)) {
    osp_snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
  } else if (s >= (1024LL*1024LL)) {
    osp_snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
  } else if (s >= (1024LL)) {
    osp_snprintf(buf, 1000, "%.2fK",s/(1024.f));
  } else {
    osp_snprintf(buf,1000,"%zi",s);
  }
  return buf;
}

inline size_t availableDeviceMemory()
{
  size_t free, total;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free, &total));
  return free;
}

__host__ __device__
inline size_t div_up(size_t a, size_t b)
{
  return (a + b - 1) / b;
}

using namespace math;

//=========================================================
// ExaBrick model:
//=========================================================

#define EMPTY_CELL_ID ~0ull

struct ExaBrick
{
  vec3i lower;
  vec3i size;
  int   level;
};

inline std::ostream &operator<<(std::ostream &out, const ExaBrick &brick)
{
  out << "ExaBrick{lower: " << brick.lower
    << ", size: " << brick.size
    << ", level: " << brick.level << '}';
  return out;
}

struct ExaModel
{
  ExaBrick *exaBricks{nullptr};
  size_t numBricks{0ull};

  int *scalarOffsets{nullptr};

  int *cellIDs{nullptr};
  size_t numCells{0ull};

  void save();
};

void ExaModel::save() {
  std::ofstream out(g_outFileName);
  for (size_t brickID=0; brickID<numBricks; brickID+=1) {
    ExaBrick brick = exaBricks[brickID];
    vec2i range;
    range.x = scalarOffsets[brickID];
    if (brickID == numBricks-1)
      range.y = numCells;
    else
      range.y = scalarOffsets[brickID+1];
    out.write((const char *)&brick.size,sizeof(brick.size));
    out.write((const char *)&brick.lower,sizeof(brick.lower));
    out.write((const char *)&brick.level,sizeof(brick.level));
    std::vector<int> cIDs(range.y-range.x);
    for (int j=0;j<cIDs.size(); ++j) {
      int cellID=range.x+j;
      cIDs[j] = cellIDs[cellID];
    }
    out.write((const char *)cIDs.data(),sizeof(cIDs[0])*cIDs.size());
  }
}

//=========================================================
// Cell helpers
//=========================================================

struct Cell
{
  vec3i pos;
  int level;
};

inline std::ostream &operator<<(std::ostream &out, const Cell &cell)
{
  out << "Cell{pos: " << cell.pos << ", level: " << cell.level << '}';
  return out;
}

struct CellRef
{
  __device__ __host__ inline int level(size_t at) const {
    return int(mortonCode[at]>>56ull);
  }

  unsigned *cellID;
  // 8 most significant bits contain the level:
  uint64_t *mortonCode;
};

#define MAX_LEVEL 32

__global__ void computeMinMaxLevel(const Cell *cells,
                                   vec2i *minMaxLevel,
                                   size_t numCells,
                                   box3i *levelBounds)
{
  size_t cellID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (cellID >= numCells)
    return;

  __shared__ vec2i sharedMinMaxLevel;
  __shared__ box3i sharedLevelBounds[MAX_LEVEL];

  if (threadIdx.x == 0) {
    memcpy(&sharedMinMaxLevel, minMaxLevel, sizeof(sharedMinMaxLevel));
  }

  if (threadIdx.x < MAX_LEVEL) {
    memcpy(&sharedLevelBounds[threadIdx.x],
           &levelBounds[threadIdx.x],
           sizeof(sharedLevelBounds[threadIdx.x]));
  }

  __syncthreads();

  atomicMin(&sharedMinMaxLevel.x, cells[cellID].level);
  atomicMax(&sharedMinMaxLevel.y, cells[cellID].level);

  if (cells[cellID].level < MAX_LEVEL) {
    const Cell &cell = cells[cellID];
    atomicMin(&sharedLevelBounds[cell.level].lower.x, cell.pos.x);
    atomicMin(&sharedLevelBounds[cell.level].lower.y, cell.pos.y);
    atomicMin(&sharedLevelBounds[cell.level].lower.z, cell.pos.z);

    atomicMax(&sharedLevelBounds[cell.level].upper.x, cell.pos.x+(1ull<<cell.level));
    atomicMax(&sharedLevelBounds[cell.level].upper.y, cell.pos.y+(1ull<<cell.level));
    atomicMax(&sharedLevelBounds[cell.level].upper.z, cell.pos.z+(1ull<<cell.level));
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicMin(&minMaxLevel->x, sharedMinMaxLevel.x);
    atomicMax(&minMaxLevel->y, sharedMinMaxLevel.y);
  }

  if (threadIdx.x < MAX_LEVEL) {
    atomicMin(&levelBounds[threadIdx.x].lower.x, sharedLevelBounds[threadIdx.x].lower.x);
    atomicMin(&levelBounds[threadIdx.x].lower.y, sharedLevelBounds[threadIdx.x].lower.y);
    atomicMin(&levelBounds[threadIdx.x].lower.z, sharedLevelBounds[threadIdx.x].lower.z);

    atomicMax(&levelBounds[threadIdx.x].upper.x, sharedLevelBounds[threadIdx.x].upper.x);
    atomicMax(&levelBounds[threadIdx.x].upper.y, sharedLevelBounds[threadIdx.x].upper.y);
    atomicMax(&levelBounds[threadIdx.x].upper.z, sharedLevelBounds[threadIdx.x].upper.z);
  }
}

__global__ void computeLevelBounds(const Cell *cells,
                                   box3i *levelBounds,
                                   size_t numCells)
{
  size_t cellID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (cellID >= numCells)
    return;

  const Cell &cell = cells[cellID];

  atomicMin(&levelBounds[cell.level].lower.x, cell.pos.x);
  atomicMin(&levelBounds[cell.level].lower.y, cell.pos.y);
  atomicMin(&levelBounds[cell.level].lower.z, cell.pos.z);

  atomicMax(&levelBounds[cell.level].upper.x, cell.pos.x+(1ull<<cell.level));
  atomicMax(&levelBounds[cell.level].upper.y, cell.pos.y+(1ull<<cell.level));
  atomicMax(&levelBounds[cell.level].upper.z, cell.pos.z+(1ull<<cell.level));
}

// Assemble some info about the cells on each level
// that the clustering algorithms need
struct CellInfo
{
  void compute(const Cell *d_cells, size_t numCells);
  void initLevelBounds(int numLevels);

  // on the host:
  vec2i minMaxLevel{INT_MAX, 0};

  // on the device:
  vec2i *d_minMaxLevel{nullptr};
  box3i *d_levelBounds{nullptr};
};

void CellInfo::initLevelBounds(int numLevels)
{
  CUDA_SAFE_CALL(cudaFree(d_levelBounds));
  CUDA_SAFE_CALL(cudaMalloc(&d_levelBounds, sizeof(box3i)*numLevels));

  box3i *h_levelBoundsInit = (box3i *)malloc(sizeof(box3i)*numLevels);
  for (int l=0; l<numLevels; ++l) {
    h_levelBoundsInit[l] = box3i(vec3i(INT_MAX), vec3i(-INT_MAX));
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_levelBounds, h_levelBoundsInit,
    		    sizeof(box3i)*numLevels,
    		    cudaMemcpyHostToDevice));
  free(h_levelBoundsInit);
}

void CellInfo::compute(const Cell *d_cells, size_t numCells)
{
  size_t numThreads = 1024;

  CUDA_SAFE_CALL(cudaMalloc(&d_minMaxLevel, sizeof(vec2i)));
  minMaxLevel = vec2i(INT_MAX,0);
  CUDA_SAFE_CALL(cudaMemcpy(d_minMaxLevel, &minMaxLevel,
                            sizeof(minMaxLevel),
                            cudaMemcpyHostToDevice));

  initLevelBounds(MAX_LEVEL);

  computeMinMaxLevel<<<div_up(numCells, numThreads), numThreads>>>(
    d_cells, d_minMaxLevel, numCells, d_levelBounds);
  CUDA_SAFE_CALL(cudaMemcpy(&minMaxLevel, d_minMaxLevel,
                            sizeof(minMaxLevel),
                            cudaMemcpyDeviceToHost));

  int numLevels = minMaxLevel.y-minMaxLevel.x+1;
  int numLevelsBase0 = minMaxLevel.y+1;

  // more than MAX_LEVEL levels, so we need to recompute this
  // in full after all:
  if (numLevelsBase0 >= MAX_LEVEL) {
    initLevelBounds(numLevelsBase0);

    computeLevelBounds<<<div_up(numCells, numThreads), numThreads>>>(
      d_cells, d_levelBounds, numCells);
  }
}

//=========================================================
//=========================================================

__host__ __device__
inline unsigned long long morton_encode3D(unsigned long long x, unsigned long long y, unsigned long long z)
{
  auto separate_bits = [](unsigned long long n) {
    n &= 0b1111111111111111111111ull;
    n = (n ^ (n << 32)) & 0b1111111111111111000000000000000000000000000000001111111111111111ull;
    n = (n ^ (n << 16)) & 0b0000000011111111000000000000000011111111000000000000000011111111ull;
    n = (n ^ (n <<  8)) & 0b1111000000001111000000001111000000001111000000001111000000001111ull;
    n = (n ^ (n <<  4)) & 0b0011000011000011000011000011000011000011000011000011000011000011ull;
    n = (n ^ (n <<  2)) & 0b1001001001001001001001001001001001001001001001001001001001001001ull;
    return n;
  };  

  return separate_bits(x) | (separate_bits(y) << 1) | (separate_bits(z) << 2); 
}

__host__ __device__
inline vec3i morton_decode3D(unsigned long long index)
{
  auto compact_bits = [](unsigned long long n) {
    n &= 0b1001001001001001001001001001001001001001001001001001001001001001ull;
    n = (n ^ (n >>  2)) & 0b0011000011000011000011000011000011000011000011000011000011000011ull;
    n = (n ^ (n >>  4)) & 0b1111000000001111000000001111000000001111000000001111000000001111ull;
    n = (n ^ (n >>  8)) & 0b0000000011111111000000000000000011111111000000000000000011111111ull;
    n = (n ^ (n >> 16)) & 0b1111111111111111000000000000000000000000000000001111111111111111ull;
    n = (n ^ (n >> 32)) & 0b1111111111111111111111ull;
    return n;
  };  

  return { (int)compact_bits(index), (int)compact_bits(index >> 1), (int)compact_bits(index >> 2) };
}

__host__ __device__
inline void print_morton(uint64_t code, const char *str=nullptr)
{
  auto i3 = morton_decode3D(code);
  if (str)
    printf("%s morton code: %u, 3D coord: (%i,%i,%i)\n",str,(unsigned)code,i3.x,i3.y,i3.z);
  else
    printf("Morton code: %u, 3D coord: (%i,%i,%i)\n",(unsigned)code,i3.x,i3.y,i3.z);
}

namespace morton {
//      100----101
//      /|     /|
//   000-|--001 |
//    | 110--|-111
//    |/     |/
//   010----011
enum Vertex { _000, _001, _010, _011, _100, _101, _110, _111 };
enum Edge { _000_001, _010_011,
            _100_101, _110_111,
            _xxx_yyy /*all other*/};
enum Face { _000_001_010_011,
            _100_101_110_111,
            _xxx_yyy_zzz_www /*all others*/};

template <unsigned Level>
__host__ __device__ morton::Vertex vertex(uint64_t mortonCode)
{
  mortonCode >>= (Level*3);
  return (morton::Vertex)(mortonCode & 0b111);
}

template <unsigned Level>
__host__ __device__ morton::Edge edge(uint64_t code0, uint64_t code1)
{
  Vertex v0 = vertex<Level>(code0);
  Vertex v1 = vertex<Level>(code1);
  if (v0 > v1) std::swap(v0,v1);

  if (v0 == _000 && v1 == _001)
    return _000_001;
  else if (v0 == _010 && v1 == _011)
    return _010_011;
  else if (v0 == _100 && v1 == _101)
    return _100_101;
  else if (v0 == _110 && v1 == _111)
    return _110_111;

  // we only care about the horizontal or diagonal edges
  return _xxx_yyy;
}

template <unsigned Level>
__host__ __device__ morton::Face face(uint64_t code0, uint64_t code1)
{
  Vertex v0 = vertex<Level>(code0);
  Vertex v1 = vertex<Level>(code1);
  if (v0 > v1) std::swap(v0,v1);

  if (v0 == _000 && v1 == _011)
    return _000_001_010_011;
  else if (v0 == _100 && v1 == _111)
    return _100_101_110_111;

  // we only care about the upright, front-facing faces
  return _xxx_yyy_zzz_www;
}

__host__ __device__ bool sweepRight(const CellRef ref,
                                    size_t startID,
                                    size_t stride,
                                    size_t numCells)
{
  size_t first = startID + 1;
  size_t last  = first + stride;
  for (size_t i = first; i != last; ++i) {
    if (i >= numCells) return false; // out-of-range
    if (ref.mortonCode[i-1]+1 != ref.mortonCode[i]) return false;
  }
  return true; // all codes match!
}

__host__ __device__ bool sweepLeft(const CellRef ref,
                                   size_t startID,
                                   size_t stride,
                                   size_t numCells)
{
  ssize_t last = startID;
  ssize_t first  = last - ssize_t(stride);
  for (ssize_t i = first; i != last; ++i) {
    if (i < 0) return false; // out-of-range
    if (ref.mortonCode[i]+1 != ref.mortonCode[i+1]) return false;
  }
  return true; // all codes match!
}

// Find neighboring morton vertex. The neighboring vertex can only exist on the
// horizontal. Morton vertices != morton codes (!). On L0 a morton vertex is
// comprised of one morton _code_; on L1, a morton vertex comprises eight
// morton codes, and so on:
template <unsigned L>
__host__ __device__ vec2i findVertex(size_t cellID, // in local (level!) range
                                     vec2i range,
                                     const CellRef ref, // offset by level range!
                                     size_t numCells)
{
  uint64_t code = ref.mortonCode[cellID];
  Vertex v = vertex<L>(code);

  // cells per vertex on this level
  int cellsPerV = 1 * (1ull<<L)*(1ull<<L)*(1ull<<L);;

  if (v==_000 || v==_010 || v==_100 || v==_110)
    return sweepRight(ref,cellID+range.y,cellsPerV,numCells) ? vec2i{0,cellsPerV} : vec2i{0,0};
  else
    return sweepLeft(ref,cellID+range.x,cellsPerV,numCells) ? vec2i{-cellsPerV,0} : vec2i{0,0};
}

// Find neighboring morton edge. The neighboring edges can only exist on the
// vertical. See findVertex() for further context.
template <unsigned L>
__host__ __device__ vec2i findEdge(size_t cellID,
                                   vec2i range,
                                   const CellRef ref,
                                   size_t numCells)
{
  uint64_t code = ref.mortonCode[cellID];
  Edge e = edge<L>(code+range.x,code+range.y);

  // cells per edge on this level
  int cellsPerE = 2 * (1ull<<L)*(1ull<<L)*(1ull<<L);

  if (e==_000_001 || e==_100_101)
    return sweepRight(ref,cellID+range.y,cellsPerE,numCells) ? vec2i{0,cellsPerE} : vec2i{0,0};
  else
    return sweepLeft(ref,cellID+range.x,cellsPerE,numCells) ? vec2i{-cellsPerE,0} : vec2i{0,0};
}

// Find neighboring morton face. The neighboring edges is upright and oriented
// towards us/away from us but not to the sides, up, or own.  See findVertex()
// for further context.
template <unsigned L>
__host__ __device__ vec2i findFace(size_t cellID,
                                   vec2i range,
                                   const CellRef ref,
                                   size_t numCells)
{
  uint64_t code = ref.mortonCode[cellID];
  Face f = face<L>(code+range.x,code+range.y);

  // cells per face on this level
  int cellsPerF = 4 * (1ull<<L)*(1ull<<L)*(1ull<<L);

  if (f==_000_001_010_011)
    return sweepRight(ref,cellID+range.y,cellsPerF,numCells) ? vec2i{0,cellsPerF} : vec2i{0,0};
  else
    return sweepLeft(ref,cellID+range.x,cellsPerF,numCells) ? vec2i{-cellsPerF,0} : vec2i{0,0};
}

} // namespace morton

template <typename Key, typename Value>
inline void sortByKey(Key *keys, Value *values, size_t numItems)
{
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      keys, keys, values, values, numItems);

  std::cout << prettyBytes(temp_storage_bytes) << '\n';
  CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      keys, keys, values, values, numItems);

  CUDA_SAFE_CALL(cudaFree(d_temp_storage));
}

struct CellRange {
  size_t begin, end;

  __device__ bool contains(const CellRange &other) const
  { return begin <= other.begin && end >= other.end; }

  __device__ bool intersects(const CellRange &other) const
  { return (other.begin <= begin && other.end >= begin) ||
           (other.begin <= end   && other.end >= end); }

  __device__ bool overlaps(const CellRange &other) const
  { return intersects(other); }

};

__device__ bool operator==(const CellRange &a, const CellRange &b)
{
  return a.begin==b.begin && a.end==b.end;
}

__device__ bool operator>(const CellRange &a, const CellRange &b)
{
  return (a.end-a.begin) > (b.end-b.begin);
}

__device__ CellRange getCellRangeForLevel(int level,
                                          const size_t *levelOffsets,
                                          vec2i minMaxLevel,
                                          size_t numCellsTotal,
                                          size_t numLevels)
{
  level -= minMaxLevel.x;
  size_t levelBegin = levelOffsets[level];
  size_t levelEnd = levelBegin;
  int l=level+1;
  while (levelEnd<=levelBegin) {
    if (l >= numLevels) {
      levelEnd = numCellsTotal;
      break;
    }

    levelEnd = levelOffsets[l++];
  }
  return {levelBegin,levelEnd};
}

__global__ void assignMortonCodes(const Cell *cells,
                                  CellRef ref,
                                  const box3i *levelBounds,
                                  size_t numCells)
{
  size_t cellID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (cellID >= numCells)
    return;

  const Cell cell = cells[cellID];

  vec3i localPos = cell.pos - levelBounds[cell.level].lower;
  localPos.x >>= cell.level;
  localPos.y >>= cell.level;
  localPos.z >>= cell.level;

  //cellRefs[cellID].pos = cell.pos;
  //cellRefs[cellID].level = cell.level;
  ref.cellID[cellID] = (unsigned)cellID;
  ref.mortonCode[cellID] = morton_encode3D(localPos.x, localPos.y, localPos.z);

  uint64_t level(cell.level);
  level <<= 56ull; // move into 8 most significant bits
  ref.mortonCode[cellID] |= level;
}

__global__ void computeLevelOffsets(const CellRef ref,
                                    vec2i minMaxLevel,
                                    size_t *levelOffsets,
                                    size_t numCells)
{
  size_t cellID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (cellID >= numCells)
    return;

  if (cellID==0 || (ref.level(cellID-1) != ref.level(cellID))) {
    int slot = ref.level(cellID)-minMaxLevel.x;
    levelOffsets[slot] = cellID;
  }
}

__global__ void expandBricks(const CellRef ref,
                             const size_t *levelOffsets,
                             const vec2i minMaxLevel,
                             vec2i *ranges,
                             size_t numCells,
                             size_t numLevels)
{
  size_t cellID = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellID >= numCells)
    return;

  uint64_t code = ref.mortonCode[cellID];

  // only iterate over cells of *this* AMR level:
  CellRange cr
     = getCellRangeForLevel(ref.level(cellID),levelOffsets,minMaxLevel,numCells,numLevels);
  size_t levelBegin = cr.begin;
  size_t levelEnd = cr.end;

  using namespace morton;

  // apply level offsets, make poniters and offsets local:
  CellRef levelRef;
  levelRef.cellID = ref.cellID+levelBegin;
  levelRef.mortonCode = ref.mortonCode+levelBegin;
  size_t nc = levelEnd-levelBegin;
  size_t localID = cellID-levelBegin;

  vec2i &range = ranges[cellID];
  range.x = 0;
  range.y = 0;
  vec2i prevRange(range);

#define EXPAND(L)                                                                       \
  /* For each cell, check if we can find our direct horizontal                          \
     neighbor on this level: */                                                         \
  prevRange = range;                                                                    \
  range += findVertex<L>(localID,range,levelRef,nc);                                    \
  if (range.x == prevRange.x && range.y == prevRange.y) return;                         \
                                                                                        \
  /* For each cells that form a horizontal edge, check if we can find our               \
     direct vertical neighbor (also an edge) on this level: */                          \
  prevRange = range;                                                                    \
  range += findEdge<L>(localID,range,levelRef,nc);                                      \
  if (range.x == prevRange.x && range.y == prevRange.y) return;                         \
                                                                                        \
  /* For each two edges that form a face, check if we can find the                      \
     direct neighbor on htis level: */                                                  \
  prevRange = range;                                                                    \
  range += findFace<L>(localID,range,levelRef,nc);                                      \
  if (range.x == prevRange.x && range.y == prevRange.y) return;

  EXPAND(0)
  EXPAND(1)
  EXPAND(2)
  //EXPAND(3)
}

__global__ void mergeClusters(const vec2i *ranges,
                              const CellRef ref,
                              const size_t *levelOffsets,
                              const vec2i minMaxLevel,
                              size_t numCells,
                              size_t numLevels,
                              size_t *parents,
                              size_t maxSearchRadius)
{
  size_t cellID = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellID >= numCells)
    return;

  CellRange range;
  range.begin = cellID+ranges[cellID].x;
  range.end   = cellID+ranges[cellID].y;

  CellRange cr
     = getCellRangeForLevel(ref.level(cellID),levelOffsets,minMaxLevel,numCells,numLevels);
  size_t levelBegin = cr.begin;
  size_t levelEnd = cr.end;

  ssize_t searchRangeBegin
      = std::max(ssize_t(cellID)-ssize_t(maxSearchRadius), ssize_t(levelBegin));
  size_t searchRangeEnd
      = std::min(cellID+maxSearchRadius,levelEnd);

  parents[cellID] = cellID;
  for (ssize_t otherID=searchRangeBegin; otherID!=searchRangeEnd; ++otherID) {
    if (otherID==cellID) continue;
    CellRange other;
    other.begin = otherID+ranges[otherID].x;
    other.end   = otherID+ranges[otherID].y;

    size_t currID = parents[cellID]; // the current parent
    CellRange curr;
    curr.begin = currID+ranges[currID].x;
    curr.end   = currID+ranges[currID].y;

    if (other.contains(range)) {
      if (other == range && parents[cellID] == cellID) {
        // avoid circular dependency; the left-most cluster
        // (arbitrarily) wins:
        if (otherID < cellID) {
          parents[cellID] = otherID;
        }
      } else {
        // found a bigger range to fit in?

        // try to assign us to the biggest fish in the pond:
        if (other > curr)
          parents[cellID] = otherID;
        }
    } else if (other.intersects(range) && other > curr) {
#ifdef CAN_OVERLAP
      parents[cellID] = EMPTY_CELL_ID;
#else
      assert(0); // that's not supposed to happen
#endif
    }
  }
}

__global__ void findRoots(size_t *parents, size_t numCells)
{
  size_t cellID = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellID >= numCells)
    return;

  size_t n = cellID;
  for (;;) {
#ifdef CAN_OVERLAP
    if (parents[n] == EMPTY_CELL_ID) {
    // TODO: RACES?!
      parents[cellID] = EMPTY_CELL_ID;
      break;
    }
#endif

    n = parents[n];

    if (n == parents[n]) {
      parents[cellID] = n;
      break;
    }
  }
}

__global__ void makeFinalRanges(vec2i *ranges, size_t *parents, size_t numCells)
{
  size_t cellID = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellID >= numCells)
    return;

  if (parents[cellID] != cellID)
    ranges[cellID] = {INT_MAX,-INT_MAX};
  else
    ranges[cellID] = {int(cellID)+ranges[cellID].x,int(cellID)+ranges[cellID].y};
}

__global__ void assembleBricks(const vec2i *ranges,
                               const CellRef ref,
                               const box3i *levelBounds,
                               ExaBrick *bricks,
                               int *scalarOffsets,
                               size_t numBricks)
{
  size_t brickID = threadIdx.x + blockIdx.x * blockDim.x;

  if (brickID >= numBricks)
    return;

  int firstLevel = ref.level(ranges[brickID].x);
  int lastLevel  = ref.level(ranges[brickID].y);
  assert(firstLevel==lastLevel);
  int level = firstLevel;

  uint64_t firstCode = ref.mortonCode[ranges[brickID].x];
  uint64_t lastCode  = ref.mortonCode[ranges[brickID].y];
  // kill level:
  firstCode = (firstCode<<8ull)>>8ull;
  lastCode  = (lastCode<<8ull)>>8ull;

  vec3i firstPos = morton_decode3D(firstCode);
  firstPos.x <<= level;
  firstPos.y <<= level;
  firstPos.z <<= level;
  firstPos += levelBounds[level].lower;

  vec3i lastPos  = morton_decode3D(lastCode);
  lastPos.x <<= level;
  lastPos.y <<= level;
  lastPos.z <<= level;
  lastPos += levelBounds[level].lower;

  bricks[brickID].lower = firstPos;
  bricks[brickID].size  = (lastPos-firstPos) / (1ull<<firstLevel) + vec3i(1);
  bricks[brickID].level = firstLevel;

  // just initialize, need to "scan" them later:
  scalarOffsets[brickID] = ranges[brickID].y-ranges[brickID].x+1;
  assert(scalarOffsets[brickID] ==
      bricks[brickID].size.x*bricks[brickID].size.y*bricks[brickID].size.z);
}

__global__ void assembleCellIDs(const ExaBrick *bricks,
                                const int *scalarOffsets,
                                const CellRef ref,
                                const box3i *levelBounds,
                                int *cellIDs,
                                size_t numBricks,
                                size_t numCells)
{
  size_t brickID = threadIdx.x + blockIdx.x * blockDim.x;

  if (brickID >= numBricks)
    return;

  vec2i range;
  range.x = scalarOffsets[brickID];
  if (brickID == numBricks-1)
    range.y = numCells;
  else
    range.y = scalarOffsets[brickID+1];

  const ExaBrick &brick = bricks[brickID];
  int numBrickCells = range.y-range.x;
  for (int j=0; j<numBrickCells; ++j) {
    int cellID=range.x+j;
    uint64_t mortonCode = ref.mortonCode[cellID];
    mortonCode = (mortonCode<<8ull)>>8ull;
    vec3i cellPos = morton_decode3D(mortonCode);
    cellPos.x <<= brick.level;
    cellPos.y <<= brick.level;
    cellPos.z <<= brick.level;
    cellPos += levelBounds[brick.level].lower;
    vec3i pos((cellPos.x - brick.lower.x) >> brick.level,
              (cellPos.y - brick.lower.y) >> brick.level,
              (cellPos.z - brick.lower.z) >> brick.level);
    int linearIndex = pos.z * brick.size.x * brick.size.y
                      + pos.y * brick.size.x
                      + pos.x;
    linearIndex += range.x;
    cellIDs[linearIndex] = ref.cellID[cellID];
  }
}

//=========================================================
// Brick clustering algorithm based on Morton codes
//=========================================================

static ExaModel buildMorton(const Cell *cells, size_t numCells) {
  cuda::timer t, total;

  size_t numThreads = 1024;

  Cell *d_cells;
  std::cout << prettyBytes(sizeof(Cell)*numCells) << '\n';
  std::cout << "avail: " << prettyBytes(availableDeviceMemory()) << '\n';
  CUDA_SAFE_CALL(cudaMalloc(&d_cells, sizeof(Cell)*numCells));
  std::cout << "avail after malloc: " << prettyBytes(availableDeviceMemory()) << '\n';

  CUDA_SAFE_CALL(cudaMemcpy(d_cells, cells, sizeof(cells[0])*numCells,
             cudaMemcpyHostToDevice));

  std::cout << "time for malloc and copying cells: " << t.elapsed() << '\n';
  t.reset();

  //=======================================================
  // Stage 0.1: cmopute level ranges and bounds
  //=======================================================

  CellInfo info;
  info.compute(d_cells, numCells);
  vec2i h_minMaxLevel = info.minMaxLevel;
  int numLevels = h_minMaxLevel.y-h_minMaxLevel.x+1;
  int numLevelsBase0 = h_minMaxLevel.y+1;
  vec2i *d_minMaxLevel = info.d_minMaxLevel;
  box3i *d_levelBounds = info.d_levelBounds;

  //=======================================================
  // Stage 1: sort by morton codes
  //=======================================================

  CellRef ref;
  std::cout << prettyBytes(sizeof(CellRef)*numCells) << '\n';
  std::cout << "avail: " << prettyBytes(availableDeviceMemory()) << '\n';
  CUDA_SAFE_CALL(cudaMalloc(&ref.cellID, sizeof(unsigned)*numCells));
  CUDA_SAFE_CALL(cudaMalloc(&ref.mortonCode, sizeof(uint64_t)*numCells));
  std::cout << "avail after malloc: " << prettyBytes(availableDeviceMemory()) << '\n';

  std::cout << "compute cell info: " << t.elapsed() << '\n';
  t.reset();
  assignMortonCodes<<<div_up(numCells, numThreads), numThreads>>>(
    d_cells, ref, d_levelBounds, numCells);

  // free cells now; we can retrieve all that information
  // from the mortonCodes:
  CUDA_SAFE_CALL(cudaFree(d_cells));

  // Sort on device:
  sortByKey(ref.mortonCode, ref.cellID, numCells);
  std::cout << "assign/sort morton: " << t.elapsed() << '\n';
  t.reset();

  // Compute offsets so we can iterate over per-cell levels
  size_t *d_levelOffsets;
  CUDA_SAFE_CALL(cudaMalloc(&d_levelOffsets, sizeof(size_t)*numLevels));
  CUDA_SAFE_CALL(cudaMemset(d_levelOffsets, 0, sizeof(size_t)*numLevels));
  computeLevelOffsets<<<div_up(numCells, numThreads), numThreads>>>(
    ref, h_minMaxLevel, d_levelOffsets, numCells);

  std::cout << "level offset/sort: " << t.elapsed() << '\n';
  t.reset();

  //=======================================================
  // Stage 2: try to find bricks as big as possible. At
  // the beginning every cell is a brick; we greedily try
  // to include more direct neighbors on the morton curve,
  // with invariant that the resulting bricks are convex
  //=======================================================

  vec2i *d_ranges; // these are relative to cellIDs
  std::cout << prettyBytes(sizeof(vec2i)*numCells) << '\n';
  CUDA_SAFE_CALL(cudaMalloc(&d_ranges, sizeof(vec2i)*numCells));
  expandBricks<<<div_up(numCells, numThreads), numThreads>>>(
    ref, d_levelOffsets, h_minMaxLevel, d_ranges, numCells, numLevels);

  std::cout << "expand bricks: " << t.elapsed() << '\n';
  t.reset();

  //=======================================================
  // Stage 3: get rid of redundant clusters; bump smaller
  // into bigger ones
  //=======================================================

  // sync. with EXPAND() calls. EXPAND(3) corresponds to 4096 max.
  //size_t maxBrickSize{4096};
  size_t maxBrickSize{512};
  size_t *d_parents;
  CUDA_SAFE_CALL(cudaMalloc(&d_parents, sizeof(size_t)*numCells));
  // compute tentatively for each cluster with which other one (if any)
  // it wants to merge with (but don't execute the merge yet):
  mergeClusters<<<div_up(numCells, numThreads), numThreads>>>(
    d_ranges, ref, d_levelOffsets, h_minMaxLevel, numCells, numLevels,
    d_parents, maxBrickSize);

  std::cout << "merge clusters: " << t.elapsed() << '\n';
  t.reset();

  findRoots<<<div_up(numCells, numThreads), numThreads>>>(d_parents, numCells);

  std::cout << "find roots: " << t.elapsed() << '\n';
  t.reset();

  makeFinalRanges<<<div_up(numCells, numThreads), numThreads>>>(
    d_ranges, d_parents, numCells);

  auto *lastCluster = thrust::remove_if(thrust::device, d_ranges, d_ranges + numCells,
    []__device__(const vec2i r) { return r.x > r.y; });

  std::cout << "compaction: " << t.elapsed() << '\n';
  t.reset();

  size_t numBricks = lastCluster-d_ranges;

  std::cout << numCells << " cells -> " << numBricks << " bricks\n";

  //=======================================================
  // Stage 4: convert ranges to "ExaBricks"
  //=======================================================

  ExaBrick *d_exaBricks;
  CUDA_SAFE_CALL(cudaMalloc(&d_exaBricks, sizeof(ExaBrick)*numBricks));

  int *d_scalarOffsets; // (separate!) list of scalar offsets per brick
  CUDA_SAFE_CALL(cudaMalloc(&d_scalarOffsets, sizeof(int)*numBricks));

  assembleBricks<<<div_up(numBricks, numThreads), numThreads>>>(
    d_ranges, ref, d_levelBounds, d_exaBricks, d_scalarOffsets, numBricks);

  // TODO: copy from ranges (was easier this way so keeping it for now)
  thrust::exclusive_scan(thrust::device,
                         d_scalarOffsets,
                         d_scalarOffsets+numBricks,
                         d_scalarOffsets);

  int *d_cellIDs;
  CUDA_SAFE_CALL(cudaMalloc(&d_cellIDs, sizeof(int)*numCells));

  // TODO: if this turns out as bottleneck, try to parallelize
  // over all cells and not all bricks:
  assembleCellIDs<<<div_up(numBricks, numThreads), numThreads>>>(
    d_exaBricks, d_scalarOffsets, ref, d_levelBounds, d_cellIDs, numBricks, numCells);

  std::cout << "brick assembly: " << t.elapsed() << '\n';
  t.reset();

  std::cout << "total time for building clusters: " << total.elapsed() << '\n';

  //=======================================================
  // Finalize
  //=======================================================

  // Write out:
  ExaModel model;
  model.numBricks = numBricks;
  model.numCells  = numCells;

  model.exaBricks = (ExaBrick *)malloc(numBricks*sizeof(ExaBrick));
  CUDA_SAFE_CALL(cudaMemcpy(model.exaBricks, d_exaBricks, numBricks*sizeof(ExaBrick),
                            cudaMemcpyDeviceToHost));

  model.scalarOffsets = (int *)malloc(numBricks*sizeof(int));
  CUDA_SAFE_CALL(cudaMemcpy(model.scalarOffsets, d_scalarOffsets, numBricks*sizeof(int),
                            cudaMemcpyDeviceToHost));

  model.cellIDs = (int *)malloc(numCells*sizeof(int));
  CUDA_SAFE_CALL(cudaMemcpy(model.cellIDs, d_cellIDs, numCells*sizeof(int),
                            cudaMemcpyDeviceToHost));

#if DO_PRINT_BOUNDS
  //=======================================================
  // Check 1: do bounds match?
  //=======================================================

  std::vector<box3i> cellBounds(numLevelsBase0);
  std::vector<box3i> brickBounds(numLevelsBase0, box3i(vec3i(INT_MAX), vec3i(-INT_MAX)));
  CUDA_SAFE_CALL(cudaMemcpy(cellBounds.data(), d_levelBounds, sizeof(box3i)*numLevelsBase0,
                            cudaMemcpyDeviceToHost));
  std::vector<int> brickCount(numLevelsBase0,0);

  for (size_t brickID=0; brickID<numBricks; brickID+=1) {
    ExaBrick brick = model.exaBricks[brickID];
    brickCount[brick.level]++;
    box3i bb(brick.lower, brick.lower + brick.size*(1ull<<brick.level));
    brickBounds[brick.level].lower.x = std::min(brickBounds[brick.level].lower.x,bb.lower.x);
    brickBounds[brick.level].lower.y = std::min(brickBounds[brick.level].lower.y,bb.lower.y);
    brickBounds[brick.level].lower.z = std::min(brickBounds[brick.level].lower.z,bb.lower.z);
    brickBounds[brick.level].upper.x = std::max(brickBounds[brick.level].upper.x,bb.upper.x);
    brickBounds[brick.level].upper.y = std::max(brickBounds[brick.level].upper.y,bb.upper.y);
    brickBounds[brick.level].upper.z = std::max(brickBounds[brick.level].upper.z,bb.upper.z);
  }

  for (int l=0; l<numLevelsBase0; ++l) {
    std::cout << "Level: " << l << '\n';
    std::cout << "Num bricks: " << brickCount[l] << '\n';
    std::cout << "Cell bounds: " << cellBounds[l] << '\n';
    std::cout << "Brick bounds: " << brickBounds[l] << '\n';
  }
#endif

  // Cleanup:
  CUDA_SAFE_CALL(cudaFree(d_minMaxLevel));
  CUDA_SAFE_CALL(cudaFree(d_levelBounds));
  CUDA_SAFE_CALL(cudaFree(ref.cellID));
  CUDA_SAFE_CALL(cudaFree(ref.mortonCode));
  CUDA_SAFE_CALL(cudaFree(d_ranges));
  CUDA_SAFE_CALL(cudaFree(d_parents));

#if DO_EXEC_NVIDIA_SMI
  std::cout << exec("nvidia-smi") << '\n';
#endif

  CUDA_SAFE_CALL(cudaFree(d_scalarOffsets));
  CUDA_SAFE_CALL(cudaFree(d_exaBricks));
  CUDA_SAFE_CALL(cudaFree(d_cellIDs));

  return model;
}


//=========================================================
// Brick clustering algorithm injecting empty cells
//=========================================================

namespace gridProject {

struct MacroCell
{
  int mcID;
  vec3i lower; // in local coordinates (world coords divided by cellWidth)
  int level;
  vec3i upper; // in local coordinates (world coords divided by cellWidth)
};

static constexpr int MC_WIDTH = 8;

typedef int MacroCellRef;

// brick ID from pos and bounds, both in logical level grid  coords
__device__ inline size_t mcID(const vec3i pos, const box3i &bounds)
{
  vec3i mcIndex = (pos-bounds.lower) / vec3i(MC_WIDTH);

  vec3i size = bounds.size();
  vec3i numMCs;
  numMCs.x = div_up(size.x, MC_WIDTH);
  numMCs.y = div_up(size.y, MC_WIDTH);
  numMCs.z = div_up(size.z, MC_WIDTH);

  return mcIndex.x + mcIndex.y * numMCs.x
      + mcIndex.z * size_t(numMCs.x) * numMCs.y;
}

// initialize all "active bricks" to empty/inactive
__global__ void initMacroCells(MacroCell *macrocells, size_t numMCs)
{
  size_t mcID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (mcID >= numMCs)
    return;

  macrocells[mcID].mcID = -1;
  macrocells[mcID].lower = vec3i(INT_MAX);
  macrocells[mcID].level = -1;
  macrocells[mcID].upper = vec3i(INT_MIN);
}

// project cells onto MCs to determine their size
__global__ void resizeMacroCells(const Cell *cells,
                                 size_t numCells,
                                 const box3i *levelBounds,
                                 MacroCell *macrocells,
                                 size_t *mcOffsets)
{
  size_t cellID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (cellID >= numCells)
    return;

  const Cell &cell = cells[cellID];

  int cellWidth = 1ull<<cell.level;

  vec3i posLocal = cell.pos / vec3i(cellWidth);
  box3i boundsLocal(levelBounds[cell.level].lower / vec3i(cellWidth),
                    levelBounds[cell.level].upper / vec3i(cellWidth));

  size_t mcID_local = mcID(posLocal, boundsLocal);
  auto mcID = mcOffsets[cell.level] + mcID_local;

  //printf("mcID: %i (local:) %i -- posLocal: %i,%i,%i\n",
  //  (int)mcID,(int)mcID_local,posLocal.x,posLocal.y,posLocal.z);
  MacroCell &mc = macrocells[mcID];

  atomicMin(&mc.lower.x, posLocal.x);
  atomicMin(&mc.lower.y, posLocal.y);
  atomicMin(&mc.lower.z, posLocal.z);

  atomicMax(&mc.upper.x, posLocal.x);
  atomicMax(&mc.upper.y, posLocal.y);
  atomicMax(&mc.upper.z, posLocal.z);

  atomicMax(&mc.mcID, (int)mcID);
  atomicMax(&mc.level, cell.level);
}

__global__ void createMacroCellRefs(const MacroCell *macrocells,
                                    MacroCellRef *refs,
                                    size_t numMCs)
{
  size_t mcID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (mcID >= numMCs)
    return;

  const auto &mc = macrocells[mcID];
  refs[mc.mcID] = mcID;
}

__global__ void createInitialOffsets(const MacroCell *macrocells,
                                     size_t numMCs,
                                     int *scalarOffsets)
{
  size_t mcID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (mcID >= numMCs)
    return;

  const auto &mc = macrocells[mcID];

  scalarOffsets[mcID] = -1; // inval
  if (mc.lower.x <= mc.upper.x) {
    vec3i numCells = mc.upper-mc.lower+vec3i(1);
    scalarOffsets[mcID] = numCells.x*numCells.y*numCells.z;
  }
  assert(scalarOffsets[mcID] > 0);
}

__global__ void invalidateCellIDs(int *cellIDs, size_t numCellIDs)
{
  size_t cellID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (cellID >= numCellIDs)
    return;

  cellIDs[cellID] = -1;
}

__global__ void populateCellIDs(int *cellIDs,
                                const Cell *cells,
                                size_t numCells,
                                const box3i *levelBounds,
                                const MacroCell *macrocells,
                                const MacroCellRef *refs,
                                int *scalarOffsets,
                                size_t *mcOffsets)
{
  size_t cellID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (cellID >= numCells)
    return;

  const Cell &cell = cells[cellID];

  int cellWidth = 1ull<<cell.level;

  vec3i posLocal = cell.pos / vec3i(cellWidth);
  box3i boundsLocal(levelBounds[cell.level].lower / vec3i(cellWidth),
                    levelBounds[cell.level].upper / vec3i(cellWidth));

  size_t mcID_local = mcID(posLocal, boundsLocal);
  auto mcID = mcOffsets[cell.level] + mcID_local;

  //printf("%i/%i -- %i,%i,%i\n",(int)refID,(int)mcID,mcIndex.x,mcIndex.y,mcIndex.z);
  const MacroCell &mc = macrocells[refs[mcID]];
  //printf("%i %i,%i,%i\n",mc.mcID,mc.lower.x,mc.lower.y,mc.lower.z);

  // move local to *MC coords* now:
  posLocal -= mc.lower;

  vec3i nc = mc.upper-mc.lower+vec3i(1);
  int localID = posLocal.x + posLocal.y * nc.x + posLocal.z * nc.x * nc.y;

  int offset = scalarOffsets[refs[mcID]];
  auto globalID = offset + localID;

  cellIDs[globalID] = cellID;
}

__global__ void assembleBricks(const MacroCell *macrocells,
                               const MacroCellRef *refs,
                               ExaBrick *bricks,
                               size_t numBricks)
{
  size_t brickID = threadIdx.x + size_t(blockIdx.x) * blockDim.x;

  if (brickID >= numBricks)
    return;

  ExaBrick &brick = bricks[brickID];
  const MacroCell &mc = macrocells[brickID];

  brick.lower = mc.lower * vec3i(1ull << mc.level);
  brick.size  = mc.upper-mc.lower+vec3i(1);
  brick.level = mc.level;
}

} // ::gridProject

static ExaModel buildGridProject(const Cell *cells, size_t numCells) {
  using namespace gridProject;

  cuda::timer t, total;

  size_t numThreads = 1024;

  Cell *d_cells;
  std::cout << prettyBytes(sizeof(Cell)*numCells) << '\n';
  std::cout << "avail: " << prettyBytes(availableDeviceMemory()) << '\n';
  CUDA_SAFE_CALL(cudaMalloc(&d_cells, sizeof(Cell)*numCells));
  std::cout << "avail after malloc: " << prettyBytes(availableDeviceMemory()) << '\n';

  CUDA_SAFE_CALL(cudaMemcpy(d_cells, cells, sizeof(cells[0])*numCells,
             cudaMemcpyHostToDevice));

  std::cout << "time for malloc and copying cells: " << t.elapsed() << '\n';
  t.reset();

  //=======================================================
  // Stage 0.1: cmopute level ranges and bounds
  //=======================================================

  CellInfo info;
  info.compute(d_cells, numCells);
  vec2i h_minMaxLevel = info.minMaxLevel;
  int numLevels = h_minMaxLevel.y-h_minMaxLevel.x+1;
  int numLevelsBase0 = h_minMaxLevel.y+1;
  vec2i *d_minMaxLevel = info.d_minMaxLevel;
  box3i *d_levelBounds = info.d_levelBounds;

  box3i *h_levelBounds = (box3i *)malloc(sizeof(box3i)*numLevelsBase0);
  CUDA_SAFE_CALL(cudaMemcpy(h_levelBounds, d_levelBounds,
                            sizeof(box3i)*numLevelsBase0,
                            cudaMemcpyDeviceToHost));

  std::cout << "compute cell info: " << t.elapsed() << '\n';
  t.reset();
  // for (int l=0; l<numLevelsBase0; ++l) {
  //   std::cout << h_levelBounds[l] << '\n';
  // }

  vec3i *h_numMCsPerLevel = (vec3i *)malloc(sizeof(vec3i)*numLevelsBase0);
  for (int l=0; l<numLevelsBase0; ++l) {
    h_numMCsPerLevel[l] = vec3i(0);
    if (!h_levelBounds[l].empty()) {
      int cellWidth = 1ull<<l;
      vec3i size = h_levelBounds[l].size() / cellWidth;
      h_numMCsPerLevel[l].x = div_up(size.x, MC_WIDTH);
      h_numMCsPerLevel[l].y = div_up(size.y, MC_WIDTH);
      h_numMCsPerLevel[l].z = div_up(size.z, MC_WIDTH);
    }
  }

  //=======================================================
  // Stage 1: activate and resize participating MCs
  //=======================================================

  std::vector<size_t> numMCsMaxPerLevel(numLevelsBase0);
  size_t numMCsMaxTotal = 0;
  for (int l=0; l<numLevelsBase0; ++l) {
    size_t numMCsMax = h_numMCsPerLevel[l].x
                    * size_t(h_numMCsPerLevel[l].y)
                    * h_numMCsPerLevel[l].z;
    numMCsMaxTotal += numMCsMax;
    numMCsMaxPerLevel[l] = numMCsMax;
  }

  MacroCell *d_macrocells{nullptr};
  CUDA_SAFE_CALL(cudaMalloc(&d_macrocells, sizeof(MacroCell)*numMCsMaxTotal));
  initMacroCells<<<div_up(numMCsMaxTotal, numThreads), numThreads>>>(
    d_macrocells, numMCsMaxTotal);

  auto mcOffsets = numMCsMaxPerLevel;
  size_t *d_mcOffsets{nullptr};
  CUDA_SAFE_CALL(cudaMalloc(&d_mcOffsets, sizeof(size_t)*numLevelsBase0));
  CUDA_SAFE_CALL(cudaMemcpy(d_mcOffsets, mcOffsets.data(),
                            sizeof(size_t)*numLevelsBase0,
                            cudaMemcpyHostToDevice));

  std::cout << "initMacroCells/cpy offsets: " << t.elapsed() << '\n';
  t.reset();

  // bascically doing this on the GPU out of convenience;
  // these are just a handful of entries...
  thrust::exclusive_scan(thrust::device,
                         d_mcOffsets,
                         d_mcOffsets+numLevelsBase0,
                         d_mcOffsets);

  resizeMacroCells<<<div_up(numCells, numThreads), numThreads>>>(
    d_cells, numCells, d_levelBounds, d_macrocells, d_mcOffsets);

  std::cout << "resizeMacroCells: " << t.elapsed() << '\n';
  t.reset();

  //=======================================================
  // Stage 2: create slots for all MCs and cells
  //=======================================================

  // Compaction:
  auto *lastIt = thrust::remove_if(thrust::device, d_macrocells,
    d_macrocells + numMCsMaxTotal,
    [=]__device__(const MacroCell &mc) { return mc.lower.x > mc.upper.x; });

  size_t numMCs = lastIt-d_macrocells;
  std::cout << "# MCs (total): " << numMCsMaxTotal << '\n';
  std::cout << "# MCs (compact): " << numMCs << '\n';

  MacroCellRef *d_refs{nullptr};
  CUDA_SAFE_CALL(cudaMalloc(&d_refs, sizeof(MacroCellRef)*numMCsMaxTotal));

  // mapping data structure from original to uncompacted IDs:
  createMacroCellRefs<<<div_up(numMCs, numThreads), numThreads>>>(
    d_macrocells, d_refs, numMCs);

  int *d_scalarOffsets{nullptr};
  CUDA_SAFE_CALL(cudaMalloc(&d_scalarOffsets, sizeof(int)*numMCs));

  createInitialOffsets<<<div_up(numMCs, numThreads), numThreads>>>(
    d_macrocells, numMCs, d_scalarOffsets);

  std::cout << "createInitialOffsets: " << t.elapsed() << '\n';
  t.reset();

  thrust::exclusive_scan(thrust::device,
                         d_scalarOffsets,
                         d_scalarOffsets+numMCs,
                         d_scalarOffsets);

  // this is >= numCells, as the MCs can contain empty cells:
  int numCellsTotal = 0;
  CUDA_SAFE_CALL(cudaMemcpy(&numCellsTotal, d_scalarOffsets+numMCs-1,
                            sizeof(numCellsTotal), cudaMemcpyDeviceToHost));

  MacroCell lastMC;
  CUDA_SAFE_CALL(cudaMemcpy(&lastMC, d_macrocells+numMCs-1,
                            sizeof(lastMC), cudaMemcpyDeviceToHost));

  vec3i nc = lastMC.upper-lastMC.lower+vec3i(1);
  numCellsTotal += nc.x*nc.y*nc.z;
  std::cout << "numCells:" << numCells << " vs. numCellsTotal: " << numCellsTotal << '\n';

  int *d_cellIDs{nullptr};
  CUDA_SAFE_CALL(cudaMalloc(&d_cellIDs, sizeof(int)*numCellsTotal));

  invalidateCellIDs<<<div_up(numCellsTotal, numThreads), numThreads>>>(
    d_cellIDs, numCellsTotal);

  std::cout << "invalidateCellIDs: " << t.elapsed() << '\n';
  t.reset();

  populateCellIDs<<<div_up(numCells, numThreads), numThreads>>>(
    d_cellIDs, d_cells, numCells, d_levelBounds, d_macrocells, d_refs, d_scalarOffsets, d_mcOffsets);

  std::cout << "populateCellIDs: " << t.elapsed() << '\n';
  t.reset();

  std::cout << "total time for building clusters: " << total.elapsed() << '\n';

  //=======================================================
  // Stage 5: convert ranges to "ExaBricks"
  //=======================================================

  size_t numBricks = numMCs;

  ExaBrick *d_exaBricks;
  CUDA_SAFE_CALL(cudaMalloc(&d_exaBricks, sizeof(ExaBrick)*numBricks));

  assembleBricks<<<div_up(numBricks, numThreads), numThreads>>>(
    d_macrocells, d_refs, d_exaBricks, numBricks);

  ExaModel model;
  model.numBricks = numBricks;
  model.numCells  = numCellsTotal;

  model.exaBricks = (ExaBrick *)malloc(numBricks*sizeof(ExaBrick));
  CUDA_SAFE_CALL(cudaMemcpy(model.exaBricks, d_exaBricks, numBricks*sizeof(ExaBrick),
                            cudaMemcpyDeviceToHost));

  model.scalarOffsets = (int *)malloc(numBricks*sizeof(int));
  CUDA_SAFE_CALL(cudaMemcpy(model.scalarOffsets, d_scalarOffsets, numBricks*sizeof(int),
                            cudaMemcpyDeviceToHost));

  model.cellIDs = (int *)malloc(numCellsTotal*sizeof(int));
  CUDA_SAFE_CALL(cudaMemcpy(model.cellIDs, d_cellIDs, numCellsTotal*sizeof(int),
                            cudaMemcpyDeviceToHost));

#if DO_PRINT_BOUNDS
  //=======================================================
  // Check 1: do bounds match?
  //=======================================================

  std::vector<box3i> cellBounds(numLevelsBase0);
  std::vector<box3i> brickBounds(numLevelsBase0, box3i(vec3i(INT_MAX), vec3i(-INT_MAX)));
  CUDA_SAFE_CALL(cudaMemcpy(cellBounds.data(), d_levelBounds, sizeof(box3i)*numLevelsBase0,
                            cudaMemcpyDeviceToHost));
  std::vector<int> brickCount(numLevelsBase0,0);

  for (size_t brickID=0; brickID<numBricks; brickID+=1) {
    ExaBrick brick = model.exaBricks[brickID];
    brickCount[brick.level]++;
    box3i bb(brick.lower, brick.lower + brick.size*(1ull<<brick.level));
    brickBounds[brick.level].lower.x = std::min(brickBounds[brick.level].lower.x,bb.lower.x);
    brickBounds[brick.level].lower.y = std::min(brickBounds[brick.level].lower.y,bb.lower.y);
    brickBounds[brick.level].lower.z = std::min(brickBounds[brick.level].lower.z,bb.lower.z);
    brickBounds[brick.level].upper.x = std::max(brickBounds[brick.level].upper.x,bb.upper.x);
    brickBounds[brick.level].upper.y = std::max(brickBounds[brick.level].upper.y,bb.upper.y);
    brickBounds[brick.level].upper.z = std::max(brickBounds[brick.level].upper.z,bb.upper.z);
  }

  for (int l=0; l<numLevelsBase0; ++l) {
    std::cout << "Level: " << l << '\n';
    std::cout << "Num bricks: " << brickCount[l] << '\n';
    std::cout << "Cell bounds: " << cellBounds[l] << '\n';
    std::cout << "Brick bounds: " << brickBounds[l] << '\n';
  }
#endif

  // Cleanup
  free(h_levelBounds);
  free(h_numMCsPerLevel);
  CUDA_SAFE_CALL(cudaFree(d_cells));
  CUDA_SAFE_CALL(cudaFree(d_mcOffsets));
  CUDA_SAFE_CALL(cudaFree(d_macrocells));

#if DO_EXEC_NVIDIA_SMI
  std::cout << exec("nvidia-smi") << '\n';
#endif

  CUDA_SAFE_CALL(cudaFree(d_scalarOffsets));
  CUDA_SAFE_CALL(cudaFree(d_exaBricks));
  CUDA_SAFE_CALL(cudaFree(d_cellIDs));

  return model;
}


//=========================================================
//=========================================================

static bool parseCommandLine(int argc, char **argv)
{
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "-o")
      g_outFileName = argv[++i];
    else if (arg == "-mode") {
      std::string mode(argv[++i]);
      if (mode == "morton") g_mode = Morton;
      else if (mode == "projection") g_mode = Projection;
      else return false;
    }
    else if (arg[0] != '-')
      g_inFileName = arg;
    else return false;
  }

  return true;
}

static bool validateInput()
{
  if (g_inFileName.empty() || g_outFileName.empty())
    return false;

  return true;
}

static void printUsage()
{
  std::cerr << "Usage: ./app input.cells -o output.bricks\n";
}

int main(int argc, char *argv[]) {
  if (!parseCommandLine(argc, argv)) {
    printUsage();
    exit(1);
  }

  if (!validateInput()) {
    printUsage();
    exit(1);
  }

  std::ifstream cellFile(g_inFileName, std::ios::binary);
  if (!cellFile.good()) {
    std::cerr << "Cannot open cell file!\n";
    exit(-1);
  }

  cellFile.seekg(0,cellFile.end);
  size_t numCells = cellFile.tellg()/sizeof(Cell);
  cellFile.seekg(0,cellFile.beg);

  Cell *cells;
  CUDA_SAFE_CALL(cudaMallocHost(&cells,sizeof(cells[0])*numCells));
  cellFile.read((char *)cells,sizeof(cells[0])*numCells);

  ExaModel model;
  if (g_mode == Morton) model = buildMorton(cells,numCells);
  else                  model = buildGridProject(cells,numCells);
  model.save();

  // Cleanup
  CUDA_SAFE_CALL(cudaFreeHost(cells));
  free(model.exaBricks);
  free(model.scalarOffsets);
  free(model.cellIDs);
}




