/*  ==========================================================================================
    Author: Leonardo Citraro
    Company:
    Filename: wrapper.cpp
    Last modifed:   17.3.2021 by Leonardo Citraro
    Description:    Boost-Python wrappers
    ==========================================================================================
    Copyright (c) 2017 Leonardo Citraro <ldo.citraro@gmail.com>
    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify,
    merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following
    conditions:
    The above copyright notice and this permission notice shall be included in all copies
    or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    ==========================================================================================
*/
#ifndef BOOST_PYTHON_STATIC_LIB
#define BOOST_PYTHON_STATIC_LIB
#endif
#include <Python.h>
#include <boost/python.hpp>
#include <algorithm>
#include <unordered_map>
#include "Graph.h"

using namespace std;
namespace p = boost::python;

inline size_t node_key(int i, int j) { return (size_t) i << 32 | (unsigned int) j; }

Graph* init(p::list graph_as_text, int verbose) {
    int n, m; ////no of nodes, no of arcs;
    char pr_type[3]; ////problem type;
    int tail, head;
    double weight=0;
    double en_weight=0;
    double ex_weight=0;

    vector<int> edge_tails, edge_heads;
    vector<double> edge_weights;

    p::ssize_t len = p::len(graph_as_text);

    string line_inf = p::extract<string>(graph_as_text[0]);
    sscanf(line_inf.c_str(), "%*c %3s %d %d", pr_type, &n, &m);

    auto *resG = new Graph(n, m, 0, n - 1, en_weight, ex_weight);
    int edges = 0;
    int edge_id = 0;
    if (verbose>=2)
        cout << "Parsing edges: " <<endl;
    
    string line;
    for (int i=1; i<len; i++) {
        line = p::extract<string>(graph_as_text[i]);
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
    if (verbose>=2)
        cout <<"Parsing done!"<<endl;
    return resG;
}
/* there is still an issue with this 
p::list make_output(Graph* resG, vector<vector<int>> path_set, double cost_sum) {
    //r some reason path_set doesn't contain the actual (final) shortest paths.
    //It contains the information of which edge is active in the graph instead.
    //This function outputs these edges.
    //!! some edges may be flipped !!
    //The next step would be to extract the subgraph composed of only active edges and then compute all paths
    //connecting source to sink. (this is done in the python code)
    
    int i, j;
    int tail, head, edge_id;
    std::unordered_map<size_t, int> m;
    
    std::vector<bool> edge_visited;   

    for (i = 0; i < path_set.size(); i++) {
        for (j = 0; j < path_set[i].size() - 1; j++) {
            head = path_set[i][j];
            tail = path_set[i][j + 1];
            
            if (m.find(node_key(tail, head)) == m.end()) {
                // not found condition
                edge_visited.push_back(true);
                m.insert({node_key(tail, head), edge_visited.size()-1});
            }else{
                edge_id = m[node_key(tail, head)];
                edge_visited[edge_id] = !edge_visited[edge_id];    
            }
        }
    }
    
    p::list edges;
    for (i = 0; i < path_set.size(); i++) {
        for (j = path_set[i].size() - 1; j>0; j--) {
            //head = path_set[i][j];
            //tail = path_set[i][j - 1]; 
            head = path_set[i][j-1];
            tail = path_set[i][j];
            
            edge_id = m[node_key(tail, head)];
            if (edge_visited[edge_id]) {
                //edges.append(p::make_tuple(tail+1, head+1));
                edges.append(p::make_tuple(head+1, tail+1));
            }
        }
    }
            
    return edges;
}
*/
p::list make_output2(Graph* resG, vector<vector<int>> path_set, double cost_sum) {
    // this requires that the flag in muSSP-master/muSSP/Graph.cpp Graph::add_edge is set to true!

    int i, j;
    int tail, head, edge_id;
    
    std::vector<bool> edge_visited_flag(resG->num_edges_); 

    for (i = 0; i < path_set.size(); i++) {
        for (j = 0; j < path_set[i].size() - 1; j++) {
            head = path_set[i][j];
            tail = path_set[i][j + 1];
            
            int edge_idx = resG->node_id2edge_id[Graph::node_key(tail, head)];
            edge_visited_flag[edge_idx] = !edge_visited_flag[edge_idx];
        }
    }
    
    p::list edges;
    for (int i = 0; i < resG->num_edges_; i++) {
        if (edge_visited_flag[i]){
            
            head = resG->edge_tail_head[i].first;
            tail = resG->edge_tail_head[i].second;
            
            edges.append(p::make_tuple(head+1, tail+1));
        }
    }
            
    return edges;
}

p::list solve(p::list graph_as_text, int verbose) {
    
    clock_t t_start;
    clock_t t_end;

    //// reading data
    t_start = clock();
    
    std::unique_ptr<Graph> org_graph = std::unique_ptr<Graph>(init(graph_as_text, verbose));

    t_end = clock();
    long double parsing_time = t_end - t_start;

    auto *duration = new long double[10];
    for (int i = 0; i < 10; i++)
        duration[i] = 0;
    
    //// 1: remove dummy edges
    t_start = clock();
    org_graph->invalid_edge_rm();
    t_end = clock();
    duration[0] = t_end - t_start;

    ////save path and path cost
    vector<double> path_cost;
    vector<vector<int>> path_set;
    int path_num = 0;

    t_start = clock();
    //// 2: initialize shortest path tree from the DAG
    org_graph->shortest_path_dag();
    t_end = clock();
    duration[1] = duration[1] + t_end - t_start;

    path_cost.push_back(org_graph->distance2src[org_graph->sink_id_]);
    org_graph->cur_path_max_cost = -org_graph->distance2src[org_graph->sink_id_]; // the largest cost we can accept

    //// 3: convert edge cost (make all weights positive)
    t_start = clock();
    org_graph->update_allgraph_weights();
    t_end = clock();
    duration[2] = duration[2] + t_end - t_start;

    //// 8: extract shortest path
    t_start = clock();
    org_graph->extract_shortest_path();
    t_end = clock();
    duration[7] = duration[7] + t_end - t_start;

    path_set.push_back(org_graph->shortest_path);
    path_num++;

    vector<unsigned long> update_node_num;

    //// 4: find nodes for updating based on branch node
    vector<int> node_id4updating;
    t_start = clock();
    org_graph->find_node_set4update(node_id4updating);
    t_end = clock();
    duration[3] = duration[3] + t_end - t_start;

    //// 10: rebuild residual graph by flipping paths
    t_start = clock();
    org_graph->flip_path();//also erase the top sinker
    t_end = clock();
    duration[9] = duration[9] + t_end - t_start;    
    
    while (true) {
        
        //// 6: update shortest path tree based on the selected sub-graph
        t_start = clock();
        org_graph->update_shortest_path_tree_recursive(node_id4updating);
        if (verbose>=2)
            printf("Iteration #%d, updated node number  %ld \n", path_num, org_graph->upt_node_num);
        t_end = clock();
        duration[5] = duration[5] + t_end - t_start;

        //// 7: update sink node (heap)
        t_start = clock();
        org_graph->update_sink_info(node_id4updating);
        t_end = clock();
        duration[6] = duration[6] + t_end - t_start;

        update_node_num.push_back(node_id4updating.size());

        //// 8: extract shortest path
        t_start = clock();
        org_graph->extract_shortest_path();
        t_end = clock();
        duration[7] = duration[7] + t_end - t_start;

        //// test if stop
        double cur_path_cost = path_cost[path_num - 1] + org_graph->distance2src[org_graph->sink_id_];

        if (cur_path_cost > -0.0000001) {
            break;
        }

        path_cost.push_back(cur_path_cost);
        org_graph->cur_path_max_cost = -cur_path_cost;
        path_set.push_back(org_graph->shortest_path);
        path_num++;

        //// 9: update weights
        t_start = clock();
        org_graph->update_subgraph_weights(node_id4updating);
        t_end = clock();
        duration[8] = duration[8] + t_end - t_start;

        //// 4: find nodes for updating
        t_start = clock();
        org_graph->find_node_set4update(node_id4updating);
        t_end = clock();
        duration[3] = duration[3] + t_end - t_start;
        //// 10: rebuild the graph
        t_start = clock();
        org_graph->flip_path();
        t_end = clock();
        duration[9] = duration[9] + t_end - t_start;
    }

    //// out put results and time consuming
    if (verbose>=2)
        cout << "Parsing time is: " << parsing_time / CLOCKS_PER_SEC << " s" << endl;

    long double all_cpu_time = 0;
    for (int i = 0; i < 10; i++) {
        auto time_elapsed_ms = 1000.0 * duration[i] / CLOCKS_PER_SEC;
        all_cpu_time += time_elapsed_ms;
        if (verbose>=2)
            cout << "the " << i + 1 << " step used: " << time_elapsed_ms / 1000.0 << " s\n";
    }
    if (verbose>=1)
        cout << "The overall time is " << all_cpu_time / 1000.0 << " s\n\n";

    double cost_sum = 0;
    for (int i = 0; i < path_cost.size(); i++) {
        cost_sum += path_cost[i];
        if (verbose>=1)
            printf("cost path #%d %.7f\n", i+1, path_cost[i]);
    }
    if (verbose>=1)
        printf("The number of paths: %ld, total cost is %.7f, final path cost is: %.7f. #path_set %ld\n",
               path_cost.size(), cost_sum, path_cost[path_cost.size() - 1], path_set.size());

    return make_output2(org_graph.get(), path_set, cost_sum);
}

BOOST_PYTHON_MODULE(wrapper) {
    Py_Initialize();

    p::def("solve", &solve);

};
