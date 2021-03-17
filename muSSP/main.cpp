#include <iostream>
#include <fstream>
#include <stdio.h>
#include "Graph.h"
#include <ctime>
#include <numeric>
#include <algorithm>

using namespace std;

inline size_t node_key(int i, int j) { return (size_t) i << 32 | (unsigned int) j; }
/*****
 * data parsing
 * *****/
Graph init(string filename) {
    int n, m; ////no of nodes, no of arcs;
    char pr_type[3]; ////problem type;
    int tail, head;
    double weight;
    double en_weight, ex_weight;

    vector<int> edge_tails, edge_heads;
    vector<double> edge_weights;

    ifstream file(filename);

    string line_inf;
    getline(file, line_inf);
    //cout << line <<endl;
    sscanf(line_inf.c_str(), "%*c %3s %d %d", pr_type, &n, &m);

    //getline(file, line_inf);
    //cout << line <<endl;
    //sscanf(line_inf.c_str(), "%*c %4s %lf %lf", pr_type, &en_weight, &ex_weight);

    auto *resG = new Graph(n, m, 0, n - 1, en_weight, ex_weight);
    int edges = 0;
    int edge_id = 0;
    cout << "Parsing edges: " <<endl;
    for (string line; getline(file, line);) {
        switch (line[0]) {
            case 'c':                  /* skip lines with comments */
            case '\n':                 /* skip empty lines   */
            case 'n':
            case '\0':                 /* skip empty lines at the end of file */
                break;
            case 'p':
            case 'a': {
                sscanf(line.c_str(), "%*c %d %d %lf", &tail, &head, &weight);
                edges++;

                resG->add_edge(tail - 1, head - 1, edge_id, weight);
                edge_id++;
                if (edges % 10000 == 0)
                    cout << edges << endl;
                break;
            }
            default:
                break;
        }
    }
    cout <<"Parsing done!"<<endl;
    return *resG;
}

void print_solution(Graph resG, vector<vector<int>> path_set, const char *outfile_name, double cost_sum) {
    int i, j;
    int tail, head;
    bool *edge_visited_flag = new bool[resG.num_edges_];
    for (i = 0; i < resG.num_edges_; i++) {
        edge_visited_flag[i] = false;
    }
    for (i = 0; i < path_set.size(); i++) {
        for (j = 0; j < path_set[i].size() - 1; j++) {
            tail = path_set[i][j];
            head = path_set[i][j + 1];
            int edge_idx = resG.node_id2edge_id[node_key(tail, head)];
            edge_visited_flag[edge_idx] = !edge_visited_flag[edge_idx];
        }
    }
    FILE *fp;
    fp = fopen(outfile_name, "w");
    if(cost_sum<0){
        for (i = 0; i < resG.num_edges_; i++) {
            //printf("tail %d, head %d : %d \n", resG.edge_tail_head[i].first + 1, resG.edge_tail_head[i].second + 1, edge_visited_flag[i]);
            if (edge_visited_flag[i])
                fprintf(fp, "f %d %d 1\n", resG.edge_tail_head[i].first + 1, resG.edge_tail_head[i].second + 1);
            else
                fprintf(fp, "f %d %d 0\n", resG.edge_tail_head[i].first + 1, resG.edge_tail_head[i].second + 1);
        }
    }
    fclose(fp);
}

int main(int argc, char *argv[]) {
    clock_t t_start;
    clock_t t_end;

    //// reading data
    t_start = clock();
    //Graph org_graph = init("input.txt");
    //Graph org_graph = init(
    //        "input_MOT_seq07_followme_k2.txt");
    char* in_file =  argv[2];
    Graph org_graph = init(in_file);
    t_end = clock();
    long double parsing_time = t_end - t_start;

    auto *duration = new long double[10];
    for (int i = 0; i < 10; i++)
        duration[i] = 0;
    //// 1: remove dummy edges
    t_start = clock();
    org_graph.invalid_edge_rm();
    t_end = clock();
    duration[0] = t_end - t_start;

    ////save path and path cost
    vector<double> path_cost;
    vector<vector<int>> path_set;
    int path_num = 0;

    t_start = clock();
    //// 2: initialize shortest path tree from the DAG
    org_graph.shortest_path_dag();
    t_end = clock();
    duration[1] = duration[1] + t_end - t_start;

    path_cost.push_back(org_graph.distance2src[org_graph.sink_id_]);
    org_graph.cur_path_max_cost = -org_graph.distance2src[org_graph.sink_id_]; // the largest cost we can accept

    //// 3: convert edge cost (make all weights positive)
    t_start = clock();
    org_graph.update_allgraph_weights();
    t_end = clock();
    duration[2] = duration[2] + t_end - t_start;

    //// 8: extract shortest path
    t_start = clock();
    org_graph.extract_shortest_path();
    t_end = clock();
    duration[7] = duration[7] + t_end - t_start;

    path_set.push_back(org_graph.shortest_path);
    path_num++;

    vector<unsigned long> update_node_num;

    //// 4: find nodes for updating based on branch node
    vector<int> node_id4updating;
    t_start = clock();
    org_graph.find_node_set4update(node_id4updating);
    t_end = clock();
    duration[3] = duration[3] + t_end - t_start;

    //// 10: rebuild residual graph by flipping paths
    t_start = clock();
    org_graph.flip_path();//also erase the top sinker
    t_end = clock();
    duration[9] = duration[9] + t_end - t_start;
    while (true) {
        //// 6: update shortest path tree based on the selected sub-graph
        t_start = clock();
        org_graph.update_shortest_path_tree_recursive(node_id4updating);
        printf("Iteration #%d, updated node number  %ld \n", path_num, org_graph.upt_node_num);
        t_end = clock();
        duration[5] = duration[5] + t_end - t_start;

        //// 7: update sink node (heap)
        t_start = clock();
        org_graph.update_sink_info(node_id4updating);
        t_end = clock();
        duration[6] = duration[6] + t_end - t_start;

        update_node_num.push_back(node_id4updating.size());

        //// 8: extract shortest path
        t_start = clock();
        org_graph.extract_shortest_path();
        t_end = clock();
        duration[7] = duration[7] + t_end - t_start;

        //// test if stop
        double cur_path_cost = path_cost[path_num - 1] + org_graph.distance2src[org_graph.sink_id_];

        if (cur_path_cost > -0.0000001) {
            break;
        }

        path_cost.push_back(cur_path_cost);
        org_graph.cur_path_max_cost = -cur_path_cost;
        path_set.push_back(org_graph.shortest_path);
        path_num++;

        //// 9: update weights
        t_start = clock();
        org_graph.update_subgraph_weights(node_id4updating);
        t_end = clock();
        duration[8] = duration[8] + t_end - t_start;

        //// 4: find nodes for updating
        t_start = clock();
        org_graph.find_node_set4update(node_id4updating);
        t_end = clock();
        duration[3] = duration[3] + t_end - t_start;
        //// 10: rebuild the graph
        t_start = clock();
        org_graph.flip_path();
        t_end = clock();
        duration[9] = duration[9] + t_end - t_start;
    }

    //// out put results and time consuming
    cout << "Parsing time is: " << parsing_time / CLOCKS_PER_SEC << " s" << endl;

    long double all_cpu_time = 0;
    for (int i = 0; i < 10; i++) {
        auto time_elapsed_ms = 1000.0 * duration[i] / CLOCKS_PER_SEC;
        all_cpu_time += time_elapsed_ms;
        cout << "the " << i + 1 << " step used: " << time_elapsed_ms / 1000.0 << " s\n";
    }
    cout << "The overall time is " << all_cpu_time / 1000.0 << " s\n\n";

    //// start validation
    if (0) {
        
        cout << "1 val" << endl;
        
        double cost_sum = 0, cost_sum_recalculate = 0;
        for (auto &&i : path_cost) {
            cost_sum += i;
        }
        
        cout << "2 val" << endl;
        
        for (auto &&tmpPath:path_set) {
            double tmp_path_cost = 0;
            int tmp_edge_id;
            for (int j = 0; j < tmpPath.size() - 1; j++) {
                cout << j << endl;
                cout << org_graph.node_id2edge_id.size() << "--" << tmpPath.size() << endl;
                tmp_edge_id = org_graph.node_id2edge_id[node_key(tmpPath[j + 1], tmpPath[j])];
                tmp_path_cost += org_graph.edge_org_weights[tmp_edge_id];
                org_graph.edge_org_weights[tmp_edge_id] *= -1;
            }
            cost_sum_recalculate += tmp_path_cost;
        }
        
        cout << "3 val" << endl;
        
        unsigned long total_upt_node_num = 0;
        for (auto &&i : update_node_num) {
            total_upt_node_num += i;
        }
        printf("The number of paths: %ld, total cost is %.7f, real-cost %.7f, final cost is: %.7f.\n",
               path_cost.size(), cost_sum, cost_sum_recalculate, path_cost[path_cost.size() - 1]);
    }
    double cost_sum = 0;
    for (int i = 0; i < path_cost.size(); i++) {
        cost_sum += path_cost[i];
        printf("cost path #%d %.7f\n", i+1, path_cost[i]);
    }
    printf("The number of paths: %ld, total cost is %.7f, final path cost is: %.7f. #path_set %ld\n",
           path_cost.size(), cost_sum, path_cost[path_cost.size() - 1], path_set.size());

    /*********write detailed flow to txt********/
    print_solution(org_graph, path_set, argv[3], cost_sum);
    
    return 0;
}
