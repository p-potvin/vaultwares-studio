/* Validate a .nvdb with NVIDIA's own PNanoVDB reader rather than ours.
 *
 * PNanoVDB.h is the portable C99/HLSL reference reader shipped with OpenVDB.
 * Using it here is the point: a file that our writer and our reader agree on
 * proves only that the two agree. This is an independent implementation by the
 * format's authors, so if it walks the tree and returns the right values, the
 * layout is right.
 *
 * usage: pnvalidate file.nvdb [x y z]...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PNANOVDB_C
#include "PNanoVDB.h"

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: pnvalidate file.nvdb [x y z]...\n"); return 2; }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* raw = (unsigned char*)malloc((size_t)n);
    if (!raw || fread(raw, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);

    /* FileHeader 16 B, then FileMetaData 176 B, then the name, then the grid
       buffer PNanoVDB wants. nameSize sits at metadata offset 136:
       4*8 sizes + 2*4 type/class + 6*8 worldBBox + 6*4 indexBBox + 3*8 voxelSize
       = 32 + 8 + 48 + 24 + 24. */
    unsigned int nameSize = 0;
    memcpy(&nameSize, raw + 16 + 136, 4);
    long gridOff = 16 + 176 + (long)nameSize;
    printf("file=%ld bytes nameSize=%u gridOffset=%ld\n", n, nameSize, gridOff);

    pnanovdb_buf_t buf = pnanovdb_make_buf((pnanovdb_uint32_t*)(raw + gridOff),
                                          (pnanovdb_uint64_t)((n - gridOff) / 4));
    pnanovdb_grid_handle_t grid = { pnanovdb_address_null() };
    pnanovdb_uint64_t magic = pnanovdb_grid_get_magic(buf, grid);
    printf("gridMagic=0x%llx\n", (unsigned long long)magic);
    /* NANOVDB_MAGIC_MASK ignores the trailing version digit of the magic. */
    if ((magic & 0x00FFFFFFFFFFFFFFull) != 0x004244566f6e614eull) {
        printf("MAGIC FAIL\n");
        return 1;
    }

    printf("gridType=%u gridClass=%u\n",
           pnanovdb_grid_get_grid_type(buf, grid),
           pnanovdb_grid_get_grid_class(buf, grid));
    printf("voxelSize=%g %g %g\n",
           pnanovdb_grid_get_voxel_size(buf, grid, 0),
           pnanovdb_grid_get_voxel_size(buf, grid, 1),
           pnanovdb_grid_get_voxel_size(buf, grid, 2));

    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    printf("nodes leaf=%u lower=%u upper=%u voxels=%llu\n",
           pnanovdb_tree_get_node_count_leaf(buf, tree),
           pnanovdb_tree_get_node_count_lower(buf, tree),
           pnanovdb_tree_get_node_count_upper(buf, tree),
           (unsigned long long)pnanovdb_tree_get_voxel_count(buf, tree));

    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
    pnanovdb_coord_t bmin = pnanovdb_root_get_bbox_min(buf, root);
    pnanovdb_coord_t bmax = pnanovdb_root_get_bbox_max(buf, root);
    printf("bbox=%d %d %d .. %d %d %d\n", bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);

    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(&acc, root);
    for (int i = 2; i + 2 < argc; i += 3) {
        pnanovdb_coord_t ijk;
        ijk.x = atoi(argv[i]); ijk.y = atoi(argv[i + 1]); ijk.z = atoi(argv[i + 2]);
        pnanovdb_address_t a = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, &acc, &ijk);
        printf("value %d %d %d = %.9g\n", ijk.x, ijk.y, ijk.z,
               pnanovdb_read_float(buf, a));
    }
    free(raw);
    return 0;
}
