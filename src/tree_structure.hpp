/*
 * Decision Tree Structure
 *
 * Author: Arif Qodari
 *
 */

#ifndef TREE_STRUCTURE_HPP
#define TREE_STRUCTURE_HPP

#include "opencv2/core/core.hpp"

#define RF_NUM_TREES 10
#define RF_FEAT_DIM 32
#define RF_MIN_NODE_SAMPLES 2
#define RF_MAX_TREE_DEPTH 15
#define RF_NUM_TESTS 2000
#define RF_NUM_TR_TESTS 20

namespace rf
{
    // define data structure for features
    struct sample {
        int img_index;
        cv::Vec3f label;
        cv::Vec<float, RF_FEAT_DIM> features;
    };

    // define leaf vote
    struct leaf_vote
    {
        cv::Vec3f mean;
        cv::Matx33f cov;

        // constructor
        leaf_vote() {};
        leaf_vote(cv::Vec3f m, cv::Matx33f c) : mean(m), cov(c) {};
    };

    struct node_queue_entry
    {
        int node_index;
        int node_depth;
        std::vector<int> left_child_indices;
        std::vector<int> right_child_indices;

        // constructor
        node_queue_entry(int ni, int nd, std::vector<int> lci, std::vector<int> rci) :
            node_index(ni), node_depth(nd), left_child_indices(lci), right_child_indices(rci) {};
    };
}

#endif
