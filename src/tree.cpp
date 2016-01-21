/*
 * Decision Tree
 *
 * Author: Arif Qodari
 *
 */

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"

#include <iostream>
#include <queue>

#include "tree_structure.hpp"
#include "tree.hpp"

using std::cout;
using std::clog;
using std::cerr;
using std::endl;

namespace rf
{
    ////////////////////////////////////////////////////////////////////////
    // Class Tree Node Implementation
    ////////////////////////////////////////////////////////////////////////


    /*
     * predict function
     * param[in]: data sample
     * return: result of the test
     *
     */
    int node::predict(const rf::sample& sample)
    {
        assert(m_channel >= 0);

        int test = (sample.features[m_channel] >= (float)m_threshold);
        return test;
    }


    // custom sort and split function
    bool sort_scores (const std::pair<int, float>& a, const std::pair<int, float>& b) { return (a.second < b.second); }
    bool split_scores (const std::pair<int, float>& a, const float& b) { return (a.second < b); }

    /*
     * compute regression entropy
     * param[in]: data matrix
     * param[out]: log of determinant
     *
     */
    float node::reg_entropy(const cv::Mat& data)
    {
        assert(data.rows > 0);

        // compute mean and covariance matrix
        cv::Mat mean_mat, cov_mat;
        cv::calcCovarMatrix(data, cov_mat, mean_mat, CV_COVAR_NORMAL|CV_COVAR_SCALE|CV_COVAR_ROWS, CV_32F);

        // compute determinant of covariance matrix and
        // verify that determinant of covariance matrix is always positive
        float det = (float)cv::determinant(cov_mat);
        if (det < 1e-6)
        {
            det = 0.f;
            return 0.f;
        }

        assert(det > 0);
        return log(det);
    }


    /*
     * learn node
     * param[in]: set of data samples, set of data indices
     * param[out]: set of left and right child data indices
     * return: true|false optimum split found
     *
     */
    bool node::learn(const std::vector<rf::sample>& samples, const std::vector<int>& sample_indices,
            std::vector<int>& left_child_indices, std::vector<int>& right_child_indices)
    {
        float optimum_objective_val = FLT_MIN;
        bool optimum_found = false;
        cv::RNG rng(time(NULL));

        // iterate through number of candidate splits / binary test
        for (int i = 0; i < RF_NUM_TESTS; ++i)
        {
            // generate candidate feature channel index
            int channel = rng.uniform(0, RF_FEAT_DIM);

            std::vector< std::pair<int, float> > scores(sample_indices.size());
            cv::Mat root_labels(sample_indices.size(), 3, CV_32FC1);
            float * p_root = root_labels.ptr<float>();

            // evaluate all samples
            for (int j = 0; j < sample_indices.size(); ++j)
            {
                scores[j].first = sample_indices[j];
                scores[j].second = samples[sample_indices[j]].features[channel];
                p_root[j*3] = samples[sample_indices[j]].label[0];
                p_root[j*3+1] = samples[sample_indices[j]].label[1];
                p_root[j*3+2] = samples[sample_indices[j]].label[2];
            }

            // compute entropy before split
            float root_entropy = reg_entropy(root_labels);

            // get minimum and maximum scores
            std::sort(scores.begin(), scores.end(), sort_scores);
            float min_test_val = scores.front().second;
            float max_test_val = scores.back().second;

            // iterate through candidate thresholds
            for (int j = 0; j < RF_NUM_TR_TESTS; ++j)
            {
                // generate candidate threshold
                float threshold = rng.uniform(min_test_val, max_test_val);

                // split the samples
                // do not allow all samples go to only one child (left or right)
                std::vector< std::pair<int, float> >::const_iterator split_it;
                split_it = std::lower_bound(scores.begin(), scores.end(), threshold, split_scores);
                if ((split_it - scores.begin()) == 0 || (scores.end() - split_it) == 0)
                    continue;


                // init children indices
                std::vector<int> left_sample_indices(split_it - scores.begin(), -1);
                std::vector<int> right_sample_indices(scores.end() - split_it, -1);
                assert(left_sample_indices.size() > 0);
                assert(right_sample_indices.size() > 0);
                assert(left_sample_indices.size() + right_sample_indices.size() == sample_indices.size());


                cv::Mat left_labels(left_sample_indices.size(), 3, CV_32FC1);
                float * p_left = left_labels.ptr<float>();

                // send split result to left child
                std::vector< std::pair<int, float> >::const_iterator it_scores = scores.begin();
                for (int k = 0; k < left_sample_indices.size(); ++k, ++it_scores)
                {
                    int idx = (*it_scores).first;
                    left_sample_indices[k] = idx;
                    p_left[k*3] = samples[idx].label[0];
                    p_left[k*3+1] = samples[idx].label[1];
                    p_left[k*3+2] = samples[idx].label[2];
                }

                // compute left entropy after split
                float left_entropy = reg_entropy(left_labels);
                float left_ratio = (float)left_sample_indices.size() / (float)sample_indices.size();


                cv::Mat right_labels(right_sample_indices.size(), 3, CV_32FC1);
                float * p_right = right_labels.ptr<float>();

                // send split result to right child
                for (int k = 0; k < right_sample_indices.size(); ++k, ++split_it)
                {
                    int idx = (*split_it).first;
                    right_sample_indices[k] = idx;
                    p_right[k*3] = samples[idx].label[0];
                    p_right[k*3+1] = samples[idx].label[1];
                    p_right[k*3+2] = samples[idx].label[2];
                }

                // compute right entropy after split
                float right_entropy = reg_entropy(right_labels);
                float right_ratio = (float)right_sample_indices.size() / (float)sample_indices.size();
                assert(left_ratio + right_ratio == 1.f);


                // evaluate information gain after split
                float objective_val = (root_entropy - (left_ratio * left_entropy) - (right_ratio * right_entropy));

                // update current objective measure
                // and best split parameter
                if (objective_val > optimum_objective_val)
                {
                    optimum_found = true;
                    optimum_objective_val = objective_val;

                    // save best split parameter
                    m_threshold = threshold;
                    m_channel = channel;

                    // save left and right children sample indices
                    left_child_indices = left_sample_indices;
                    right_child_indices = right_sample_indices;
                }
            }
        }

        // verify the result
        if (optimum_found)
        {
            assert(left_child_indices.size() > 0);
            assert(right_child_indices.size() > 0);
            assert(left_child_indices.size() + right_child_indices.size() == sample_indices.size());
        }

        return optimum_found;
    }


    /*
     * compute probability distribution in leaf node
     * param[in]: set of data samples, set of data indices
     *
     */
    void leaf::compute_leaf(const std::vector<rf::sample>& samples, const std::vector<int>& sample_indices)
    {
        assert(sample_indices.size() > 0);

        cv::Mat data(sample_indices.size(), 3, CV_32FC1);
        float * p_data = data.ptr<float>();

        // evaluate all samples
        for (int i = 0; i < sample_indices.size(); ++i)
        {
            p_data[i*3] = samples[sample_indices[i]].label[0];
            p_data[i*3+1] = samples[sample_indices[i]].label[1];
            p_data[i*3+2] = samples[sample_indices[i]].label[2];
        }

        // compute mean and covariance matrix
        cv::Mat mean_mat, cov_mat;
        cv::calcCovarMatrix(data, cov_mat, mean_mat, CV_COVAR_NORMAL|CV_COVAR_SCALE|CV_COVAR_ROWS, CV_32F);
        m_mean = mean_mat;
        m_cov = cov_mat;
    }


    /*
     * predict function for testing
     * param[in]: set of features and set of samples
     *
     */
    rf::leaf_vote tree::predict(const rf::sample& sample)
    {
        assert(m_tree_nodes.size() > 0);

        // start from root node
        int node_index = 0;

        // from root until leaf
        while (!m_tree_nodes[node_index]->is_leaf())
        {
            // if 1 go to right, otherwise left
            if (m_tree_nodes[node_index]->predict(sample))
                node_index = m_tree_nodes[node_index]->right_child_index();
            else
                node_index = m_tree_nodes[node_index]->left_child_index();

            assert(node_index > 0);
        }

        return rf::leaf_vote(
                m_tree_nodes[node_index]->mean(),
                m_tree_nodes[node_index]->cov()
                );
    }


    ////////////////////////////////////////////////////////////////////////
    // Class Tree Implementation
    ////////////////////////////////////////////////////////////////////////


    /*
     * add node into the tree
     * param[in]: depth index, set of data samples, set of data indices, node
     * queue
     *
     */
    void tree::add_node(const int& depth_index, const std::vector<rf::sample>& samples,
            std::vector<int>& sample_indices, std::queue<node_queue_entry>& node_queue)
    {
        int node_index = m_tree_nodes.size();
        bool optimum_found = false;

        if (sample_indices.size() >= RF_MIN_NODE_SAMPLES && depth_index < RF_MAX_TREE_DEPTH)
        {
            std::vector<int> left_child_indices, right_child_indices;
            std::unique_ptr<rf::tree_node> cnode(new rf::node(false, depth_index));

            // learn new non-leaf node
            optimum_found = cnode->learn(samples, sample_indices, left_child_indices, right_child_indices);

            if (optimum_found)
            {
                m_tree_nodes.push_back(std::move(cnode));

                // add the valid node to node queue
                node_queue_entry new_entry(node_index, depth_index, left_child_indices, right_child_indices);
                node_queue.push(new_entry);

                assert(!(m_tree_nodes.back()->is_leaf()));
            }
        }

        // create leaf when no optimum split found or reach max depth
        // or min samples left
        if (!optimum_found)
        {
            m_tree_nodes.push_back(std::unique_ptr<rf::tree_node>(new rf::leaf(true, depth_index)));
            m_tree_nodes.back()->compute_leaf(samples, sample_indices);

            assert(m_tree_nodes.back()->is_leaf());
        }

        // clear all the indices
        sample_indices.clear();

        assert(m_tree_nodes.size() == node_index + 1);
    }


    /*
     * growing tree
     * param[in]: set of data samples
     *
     */
    void tree::grow(const std::vector<rf::sample>& samples)
    {
        int depth_index = 0;
        std::queue<node_queue_entry> node_queue;

        // root node will process all incoming samples
        // create initial sample indices
        std::vector<int> sample_indices(samples.size());
        for (int i = 0; i < samples.size(); ++i)
            sample_indices[i] = i;

        // process root node
        clog << "..";
        add_node(depth_index, samples, sample_indices, node_queue);

        while (!node_queue.empty())
        {
            // update depth index
            depth_index = node_queue.front().node_depth + 1;

            // set children indices
            m_tree_nodes[node_queue.front().node_index]->set_left_child_index(m_tree_nodes.size());
            m_tree_nodes[node_queue.front().node_index]->set_right_child_index(m_tree_nodes.size()+1);

            // get sample indices for left child
            sample_indices = node_queue.front().left_child_indices;

            // process left child
            clog << "..";
            add_node(depth_index, samples, sample_indices, node_queue);

            // get sample indices for right child
            sample_indices = node_queue.front().right_child_indices;

            // process right child
            clog << "..";
            add_node(depth_index, samples, sample_indices, node_queue);

            // release parent node from queue
            node_queue.pop();
        }

        clog << endl;

        assert(depth_index <= RF_MAX_TREE_DEPTH);
        assert(m_tree_nodes.size() > 0);
    }


    /*
     * save tree model
     * param[in]: model path to store data
     *
     */
    void tree::save(const std::string& filename)
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);

        // write tree nodes
        fs << "nodes" << "[";
        for (int i = 0; i < m_tree_nodes.size(); ++i)
        {
            fs << "{";
            fs << "is_leaf" << m_tree_nodes[i]->is_leaf();
            fs << "depth" << m_tree_nodes[i]->node_depth();
            if (m_tree_nodes[i]->is_leaf())
            {
                fs << "mean" << m_tree_nodes[i]->mean();
                fs << "cov" << cv::Mat(m_tree_nodes[i]->cov());
            }
            else
            {
                fs << "channel" << m_tree_nodes[i]->channel();
                fs << "threshold" << m_tree_nodes[i]->threshold();
                fs << "left_child_index" << m_tree_nodes[i]->left_child_index();
                fs << "right_child_index" << m_tree_nodes[i]->right_child_index();
            }
            fs << "}";
        }
        fs << "]";
        fs.release();
    }


    /*
     * load tree model
     * param[in]: model path to store data
     *
     */
    void tree::load(const std::string& filename)
    {
        assert(m_tree_nodes.size() == 0);

        cv::FileStorage fs(filename, cv::FileStorage::READ);
        cv::FileNode p_nodes = fs["nodes"];
        m_tree_nodes.resize(p_nodes.size());

        for (int i = 0; i < p_nodes.size(); ++i)
        {
            bool is_leaf;
            int node_depth;
            int left_child_index;
            int right_child_index;
            int channel;
            float threshold;
            cv::Vec3f mean;
            cv::Mat temp_cov;

            p_nodes[i]["is_leaf"] >> is_leaf;
            p_nodes[i]["depth"] >> node_depth;

            assert(node_depth >= 0);

            if (!is_leaf)
            {
                p_nodes[i]["left_child_index"] >> left_child_index;
                p_nodes[i]["right_child_index"] >> right_child_index;
                p_nodes[i]["threshold"] >> threshold;
                p_nodes[i]["channel"] >> channel;

                assert(left_child_index >= 0);
                assert(right_child_index >= 0);
                assert(channel >= 0);

                m_tree_nodes[i] = std::unique_ptr<rf::tree_node>(new rf::node(is_leaf, node_depth,
                            left_child_index, right_child_index,
                            threshold, channel));
            }
            else
            {
                p_nodes[i]["mean"] >> mean;
                p_nodes[i]["cov"] >> temp_cov;
                cv::Matx33f cov(temp_cov);

                m_tree_nodes[i] = std::unique_ptr<rf::tree_node>(new rf::leaf(is_leaf, node_depth,
                            mean, cov));
            }
        }

        fs.release();
        assert(m_tree_nodes.size() > 0);
    }
}
