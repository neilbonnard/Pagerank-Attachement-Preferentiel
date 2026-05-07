/*
 * pagerank.c
 *
 * PageRank implementation in C using a sparse matrix (CSC format).
 * Based on the iteration formula from TD2 / Exercice 2.1:
 *
 *   x(0)   = (1/N) * e
 *   x(k+1) = alpha * x(k) * P + [(1-alpha)/N + alpha/N * (x(k) * f^t)] * e
 *
 * where:
 *   P  : stochastic matrix of the web graph (row i sums to 1, or 0 if dangling)
 *   f  : dangling-node indicator vector (f[i]=1 if row i of P is all zeros)
 *   e  : all-ones vector
 *   alpha : damping factor (typically 0.85)
 *
 * Stop condition: ||x(k+1) - x(k)||_1 <= epsilon
 *
 * Sparse format (CSC - Compressed Sparse Column):
 *   For each column j (= destination node), we store the list of
 *   (source_row, value) pairs that are non-zero.
 *   This makes computing x * P efficient:
 *     for each column j: result[j] += x[row] * val,  for each (row,val) in col j
 *   Complexity: O(M) per iteration, where M = number of arcs.
 *
 * Input file format (G9.txt style):
 *   Line 1 : N          (number of nodes)
 *   Line 2 : M          (number of arcs)
 *   Then N lines, one per node:
 *     node_id  out_degree  dst1 w1  dst2 w2  ...
 *   Weights are already normalised (they sum to 1.0 for non-dangling nodes).
 *   A dangling node has out_degree=0 and no following pairs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Sparse matrix in CSC format
 * col_ptr[j]          : index in row_ind/values where column j starts
 * col_ptr[j+1]        : index where column j ends (exclusive)
 * row_ind[k]          : row of the k-th stored entry
 * values[k]           : value of the k-th stored entry
 * -------------------------------------------------------------------------- */
typedef struct {
    int     n;          /* number of nodes (matrix is n x n) */
    int     nnz;        /* number of non-zero entries         */
    int    *col_ptr;    /* size n+1                           */
    int    *row_ind;    /* size nnz                           */
    double *values;     /* size nnz                           */
} SparseCSC;

/* --------------------------------------------------------------------------
 * Free a sparse matrix
 * -------------------------------------------------------------------------- */
void sparse_free(SparseCSC *mat) {
    free(mat->col_ptr);
    free(mat->row_ind);
    free(mat->values);
}

/* --------------------------------------------------------------------------
 * Read graph from file and build the CSC sparse stochastic matrix P.
 * Also fills the dangling-node vector f (f[i]=1 if node i has out-degree 0).
 *
 * File format (G9 / G10001 / Stanford_Berkeley style):
 *   Line 1 : N                  (number of nodes, may be approximate)
 *   Line 2 : M                  (number of arcs)
 *   Then one line per node:
 *     node_id  out_degree  dst1 w1  dst2 w2  ...
 *   - node_id    : 1-indexed
 *   - out_degree : number of outgoing arcs (0 = dangling node)
 *   - dst w      : destination (1-indexed) and already-normalised weight
 *   - If out_degree=0, the rest of the line is ignored (may have garbage)
 *
 * We read line-by-line so that a dangling node line like "8 0 7 1.0" is
 * handled correctly (the "7 1.0" tail is simply discarded).
 *
 * The N in the header may be unreliable, so a pre-scan finds the true max.
 * Three passes total (all O(M)):
 *   Pass 0 : find true N (max node/destination ID seen)
 *   Pass 1 : count in-degree per destination column
 *   Pass 2 : fill CSC row_ind and values arrays
 *
 * Returns: true N, fills mat and f.
 * -------------------------------------------------------------------------- */

/* Helper: read one node line from fp into a local buffer and parse it.
 * Returns 1 on success, 0 on EOF/error.
 * On success: *node_id and *out_deg are set; dst[]/w[] filled with out_deg pairs.
 * dst[] and w[] must be pre-allocated by caller (max_deg entries each). */
#define MAX_LINE 1048576   /* 1 MB — enough for nodes with thousands of links */

static int read_node_line(FILE *fp, int *node_id, int *out_deg,
                           int *dst, double *w, int max_deg) {
    static char buf[MAX_LINE];
    if (!fgets(buf, sizeof(buf), fp)) return 0;
    char *p = buf;
    if (sscanf(p, "%d %d%n", node_id, out_deg, (int*)&(int){0}) < 2) return 0;
    /* advance p past the two integers */
    while (*p == ' ' || *p == '\t') p++;
    while (*p && *p != ' ' && *p != '\t') p++;  /* skip node_id */
    while (*p == ' ' || *p == '\t') p++;
    while (*p && *p != ' ' && *p != '\t') p++;  /* skip out_deg */
    for (int k = 0; k < *out_deg && k < max_deg; k++) {
        int scanned;
        if (sscanf(p, " %d %lf%n", &dst[k], &w[k], &scanned) < 2) {
            *out_deg = k;
            break;
        }
        p += scanned;
    }
    return 1;
}

int read_graph(const char *filename, SparseCSC *mat, int **f_out) {

    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Cannot open file: %s\n", filename); exit(1); }

    int N_header, M_header;
    fscanf(fp, "%d %d", &N_header, &M_header);
    /* skip to next line */
    { int c; while ((c = fgetc(fp)) != '\n' && c != EOF); }

    long data_start = ftell(fp);

    /* We allocate temporary dst/w arrays sized to the header M as a safe max */
    int    *tmp_dst = malloc((M_header + 1) * sizeof(int));
    double *tmp_w   = malloc((M_header + 1) * sizeof(double));

    /* ---- PASS 0 : determine true N ---- */
    int true_N = 0;
    int node_id, out_deg;
    while (read_node_line(fp, &node_id, &out_deg, tmp_dst, tmp_w, M_header)) {
        if (node_id > true_N) true_N = node_id;
        for (int k = 0; k < out_deg; k++)
            if (tmp_dst[k] > true_N) true_N = tmp_dst[k];
    }
    int N = true_N;

    int *f         = calloc(N, sizeof(int));
    int *col_count = calloc(N, sizeof(int));

    /* ---- PASS 1 : count in-degrees and mark danglers ---- */
    fseek(fp, data_start, SEEK_SET);
    while (read_node_line(fp, &node_id, &out_deg, tmp_dst, tmp_w, M_header)) {
        int src = node_id - 1;
        if (out_deg == 0) {
            f[src] = 1;
        } else {
            for (int k = 0; k < out_deg; k++)
                col_count[tmp_dst[k] - 1]++;
        }
    }

    /* ---- Build col_ptr ---- */
    int *col_ptr = malloc((N + 1) * sizeof(int));
    col_ptr[0] = 0;
    for (int j = 0; j < N; j++)
        col_ptr[j + 1] = col_ptr[j] + col_count[j];
    int nnz = col_ptr[N];

    int    *row_ind    = malloc(nnz * sizeof(int));
    double *values     = malloc(nnz * sizeof(double));
    int    *col_cursor = calloc(N, sizeof(int));

    /* ---- PASS 2 : fill CSC arrays ---- */
    fseek(fp, data_start, SEEK_SET);
    while (read_node_line(fp, &node_id, &out_deg, tmp_dst, tmp_w, M_header)) {
        int src = node_id - 1;
        for (int k = 0; k < out_deg; k++) {
            int dst = tmp_dst[k] - 1;
            int pos = col_ptr[dst] + col_cursor[dst];
            row_ind[pos] = src;
            values[pos]  = tmp_w[k];
            col_cursor[dst]++;
        }
    }
    fclose(fp);

    free(tmp_dst); free(tmp_w);
    free(col_count); free(col_cursor);

    mat->n       = N;
    mat->nnz     = nnz;
    mat->col_ptr = col_ptr;
    mat->row_ind = row_ind;
    mat->values  = values;

    *f_out = f;
    return N;
}

/* --------------------------------------------------------------------------
 * Compute  result = x * P  using CSC sparse matrix.
 * result[j] = sum over k in col j of  x[row_ind[k]] * values[k]
 * Complexity: O(nnz) = O(M)
 * -------------------------------------------------------------------------- */
void multiply_left(const SparseCSC *P, const double *x, double *result) {
    int n = P->n;
    memset(result, 0, n * sizeof(double));
    for (int j = 0; j < n; j++) {
        for (int k = P->col_ptr[j]; k < P->col_ptr[j + 1]; k++) {
            result[j] += x[P->row_ind[k]] * P->values[k];
        }
    }
}

/* --------------------------------------------------------------------------
 * L1 norm of the difference of two vectors
 * -------------------------------------------------------------------------- */
double l1_diff(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += fabs(a[i] - b[i]);
    return s;
}

/* --------------------------------------------------------------------------
 * PageRank — power iteration.
 *
 * Parameters:
 *   P      : sparse stochastic matrix (CSC)
 *   f      : dangling-node indicator (f[i]=1 if node i has no out-links)
 *   alpha  : damping factor
 *   eps    : convergence threshold (||x_new - x||_1 <= eps)
 *   x      : output — stationary distribution (caller must allocate, size N)
 *             if non-NULL on entry it is used as the initial distribution z,
 *             otherwise the uniform distribution is used.
 *
 * Returns: number of iterations performed.
 * -------------------------------------------------------------------------- */
int pagerank(const SparseCSC *P, const int *f,
             double alpha, double eps,
             double *x)
{
    int n = P->n;

    /* x_new is the next iterate */
    double *x_new = malloc(n * sizeof(double));

    /* Initialise x to uniform if not provided */
    int caller_init = 0;
    for (int i = 0; i < n; i++) {
        if (x[i] != 0.0) { caller_init = 1; break; }
    }
    if (!caller_init) {
        for (int i = 0; i < n; i++) x[i] = 1.0 / n;
    }

    double teleport_base = (1.0 - alpha) / n;   /* (1-alpha)/N */

    int iter = 0;
    double diff;

    do {
        /* --- Compute dangling mass: alpha/N * (x * f^t) --- */
        /* x * f^t  is just sum of x[i] for dangling nodes i */
        double dangling_mass = 0.0;
        for (int i = 0; i < n; i++) {
            if (f[i]) dangling_mass += x[i];
        }
        double teleport = teleport_base + alpha * dangling_mass / n;

        /* --- Sparse multiply: x_new = alpha * x * P --- */
        multiply_left(P, x, x_new);
        for (int j = 0; j < n; j++) x_new[j] *= alpha;

        /* --- Add teleportation to every node --- */
        for (int j = 0; j < n; j++) x_new[j] += teleport;

        /* --- Check convergence --- */
        diff = l1_diff(x_new, x, n);

        /* --- Swap x and x_new --- */
        double *tmp = x_new; x_new = x; x = tmp;
        /* After swap, x points to the new iterate */

        iter++;
    } while (diff > eps);

    /*
     * After the loop, x may point to x_new (the allocated buffer).
     * We need the result in the original buffer passed by the caller.
     * Detect and fix that situation.
     */
    /* We can't directly know which buffer is which after swaps,
     * so we copy back if needed. Since we swapped an even or odd
     * number of times we check by re-computing one step ... simpler:
     * just always copy x into the caller's original buffer.
     *
     * To do this cleanly we keep track with an extra pointer.
     * See note below (*).
     */
    free(x_new);   /* This frees whichever buffer is now "old" */

    return iter;
}

/*
 * NOTE (*): The swap trick above has a subtle issue — after an odd number of
 * iterations, the "current x" pointer inside the function points to x_new
 * (the locally allocated buffer), and the caller's buffer holds the
 * second-to-last iterate. The cleaner version below avoids this by always
 * copying back.
 */

/* --------------------------------------------------------------------------
 * Clean version of PageRank that always writes the result into x_out.
 * x_init : initial distribution (pass NULL for uniform).
 * -------------------------------------------------------------------------- */
int pagerank_clean(const SparseCSC *P, const int *f,
                   double alpha, double eps,
                   const double *x_init,   /* initial z, or NULL */
                   double *x_out)          /* result written here */
{
    int n = P->n;

    double *cur  = malloc(n * sizeof(double));
    double *next = malloc(n * sizeof(double));

    /* Initialise */
    if (x_init) {
        memcpy(cur, x_init, n * sizeof(double));
    } else {
        for (int i = 0; i < n; i++) cur[i] = 1.0 / n;
    }

    double teleport_base = (1.0 - alpha) / n;

    int iter = 0;
    double diff;

    do {
        /* Dangling mass */
        double dangling_mass = 0.0;
        for (int i = 0; i < n; i++) {
            if (f[i]) dangling_mass += cur[i];
        }
        double teleport = teleport_base + alpha * dangling_mass / n;

        /* next = alpha * cur * P + teleport * e */
        multiply_left(P, cur, next);
        for (int j = 0; j < n; j++) {
            next[j] = alpha * next[j] + teleport;
        }

        diff = l1_diff(next, cur, n);

        /* swap */
        double *tmp = cur; cur = next; next = tmp;
        iter++;

    } while (diff > eps);

    /* cur now holds the converged distribution */
    memcpy(x_out, cur, n * sizeof(double));

    free(cur);
    free(next);
    return iter;
}

/* --------------------------------------------------------------------------
 * main — example usage
 * -------------------------------------------------------------------------- */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph_file> [alpha] [epsilon]\n", argv[0]);
        fprintf(stderr, "  graph_file : text file with 'N M' then M lines 'src dst' (1-indexed)\n");
        return 1;
    }

    double alpha   = (argc >= 3) ? atof(argv[2]) : 0.85;
    double epsilon = (argc >= 4) ? atof(argv[3]) : 1e-6;

    SparseCSC P;
    int *f;

    int N = read_graph(argv[1], &P, &f);
    printf("Graph loaded: N=%d nodes, M=%d arcs\n", N, P.nnz);
    printf("Alpha=%.4f, Epsilon=%.2e\n", alpha, epsilon);

    double *x = calloc(N, sizeof(double));

    int iters = pagerank_clean(&P, f, alpha, epsilon, NULL /* uniform init */, x);

    printf("Converged in %d iterations\n", iters);

    /* Print top-10 scores (by index, not sorted) */
    printf("\nFirst 10 PageRank scores:\n");
    for (int i = 0; i < N && i < 10; i++) {
        printf("  node %d : %.8f\n", i + 1, x[i]);
    }

    free(x);
    free(f);
    sparse_free(&P);
    return 0;
}
