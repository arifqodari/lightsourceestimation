/*
 * Decision Tree
 *
 * Author: Arif Qodari
 *
 */

#ifndef TREE_HPP
#define TREE_HPP

#include "opencv2/core/core.hpp"

#include <vector>
#include <memory>
#include <queue>

#include "tree_structure.hpp"

namespace rf
{
    ////////////////////////////////////////////////////////////////////////
    // Class Parent Tree Node Definition
    ////////////////////////////////////////////////////////////////////////
    class tree_node
    {
        protected:
        bool m_is_leaf;
        int m_node_depth;


        public:

        // constructor
        tree_node() {};
        tree_node(bool is_leaf, int node_depth) :
            m_is_leaf(is_leaf), m_node_depth(node_depth) {};

        // destructors
        ~tree_node() {};

        // getter
        bool is_leaf() {return m_is_leaf;};
        int node_depth() {return m_node_depth;};


        // virtual setter methods
        virtual void set_left_child_index(int i) {};
        virtual void set_right_child_index(int i) {};

        // virtual getter methods
        virtual int left_child_index() {return -1;};
        virtual int right_child_index() {return -1;};
        virtual float threshold() {return -1;};
        virtual int channel() {return -1;};
        virtual cv::Vec3f mean() {return cv::Vec3f();};
        virtual cv::Matx33f cov() {return cv::Matx33f();};

        // virtual public methods
        virtual int predict(const rf::sample& sample) {return -1;};
        virtual bool learn(const std::vector<rf::sample>& samples, const std::vector<int>& sample_indices,
                std::vector<int>& left_child_indices, std::vector<int>& right_child_indices) {return false;};
        virtual void compute_leaf(const std::vector<rf::sample>& samples, const std::vector<int>& sample_indices) {};
    };


    ////////////////////////////////////////////////////////////////////////
    // Class Tree Node and Leaf Definition
    ////////////////////////////////////////////////////////////////////////
    class node: public tree_node
    {
        // private members
        int m_left_child_index = -1;
        int m_right_child_index = -1;
        float m_threshold;
        int m_channel;

        // private method
        float reg_entropy(const cv::Mat& data);


        public:

        // constructor
        node() {};
        node(bool is_leaf, int node_depth) : tree_node(false, node_depth) {};
        node(
                bool is_leaf, int node_depth,
                int left_child_index, int right_child_index,
                float threshold, int channel
            ) :
            tree_node(false, node_depth),
            m_left_child_index(left_child_index), m_right_child_index(right_child_index),
            m_threshold(threshold), m_channel(channel) {};

        // destructor
        ~node() {};

        // setter
        void set_left_child_index(int i) {m_left_child_index = i;};
        void set_right_child_index(int i) {m_right_child_index = i;};

        // getter
        int left_child_index() {return m_left_child_index;};
        int right_child_index() {return m_right_child_index;};
        float threshold() {return m_threshold;};
        int channel() {return m_channel;};

        // public methods
        int predict(const rf::sample& sample);
        bool learn(const std::vector<rf::sample>& samples, const std::vector<int>& sample_indices,
                std::vector<int>& left_child_indices, std::vector<int>& right_child_indices);
    };

    class leaf: public tree_node
    {
        // private members
        cv::Vec3f m_mean;
        cv::Matx33f m_cov;


        public:

        // constructor
        leaf() {};
        leaf(bool is_leaf, int node_depth) : tree_node(true, node_depth) {};
        leaf(bool is_leaf, int node_depth, cv::Vec3f mean, cv::Matx33f cov) :
            tree_node(true, node_depth), m_mean(mean), m_cov(cov) {};

        // destructor
        ~leaf() {};

        // getter
        cv::Vec3f mean() {return m_mean;};
        cv::Matx33f cov() {return m_cov;};

        // public methods
        void compute_leaf(const std::vector<rf::sample>& samples, const std::vector<int>& sample_indices);
    };


    ////////////////////////////////////////////////////////////////////////
    // Class Tree Definition
    ////////////////////////////////////////////////////////////////////////
    class tree
    {
        // private members
        std::vector< std::unique_ptr<rf::tree_node> > m_tree_nodes;

        // private method
        void add_node(const int& depth_index, const std::vector<rf::sample>& samples,
                std::vector<int>& sample_indices, std::queue<node_queue_entry>& node_queue);


        public:

        // constructors
        tree(){};
        tree(std::vector< std::unique_ptr<rf::tree_node> > nodes)
        {
            m_tree_nodes = std::move(nodes);
        }

        // destructors
        ~tree(){};

        // getter
        int num_nodes() {return m_tree_nodes.size();};

        // public methods
        rf::leaf_vote predict(const rf::sample& sample);
        void grow(const std::vector<rf::sample>& samples);
        void save(const std::string& model_path);
        void load(const std::string& model_path);

        // deep copy object
        tree& operator=(tree &&data)
        {
            m_tree_nodes = std::move(data.m_tree_nodes);
            return *this;
        }
    };
}

#endif
