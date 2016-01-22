/*
 *
 * Author: Arif Qodari
 *
 */

#include "opencv2/core/core.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>

#include "tree_structure.hpp"
#include "tree.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using std::cout;
using std::cerr;
using std::clog;
using std::endl;


/*
 * load dataset
 * param[in]: dataset path
 * param[out]: dataset matrix
 *
 */
void
load_dataset(const fs::path& dataset_path, std::vector<rf::sample>& data)
{
    std::ifstream file(dataset_path.string());
    if (!file.is_open())
    {
        cerr << "Could not open file " << dataset_path << endl;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        rf::sample sample;

        int i = 0;
        while (std::getline(lineStream, cell, ','))
        {
            if (i == 0)
            {
                sample.img_index = std::atoi(cell.c_str());
            }
            else if (i > 0 && i < 4)
            {
                sscanf(cell.c_str(), "%f", &(sample.label[i - 1]));
            }
            // ignore column i == 4
            else if (i > 4)
            {
                sscanf(cell.c_str(), "%f", &(sample.features[i - 5]));
            }

            ++i;
        }

        data.push_back(sample);
    }
}


/*
 * sampling data with replacement
 * param[in]: dataset
 * param[out]: data samples
 *
 */
void
sampling_with_replacement(const std::vector<rf::sample>& data, std::vector<rf::sample>& samples)
{
    int n_data = data.size();
    cv::RNG rng(time(NULL));
    samples.resize(n_data);

    for (int i = 0; i < n_data; ++i)
        samples[i] = data[rng.uniform(0, n_data)];
}


/*
 * program training
 * param[in]: dataset path and model path
 *
 */
void
training(const fs::path& dataset_path, const fs::path& model_path)
{
    if (fs::exists(model_path))
    {
        cerr << "The model " << model_path << " is already exist" << endl;
        return;
    }

    // load train dataset
    std::vector<rf::sample> data;
    load_dataset(dataset_path, data);

    // sampling with replacement
    std::vector<rf::sample> samples;
    sampling_with_replacement(data, samples);
    data.clear();

    // construct tree
    cerr << "training " << model_path << endl;
    rf::tree tree;
    tree.grow(samples);

    // save tree
    tree.save(model_path.string());
}


/*
 * mean-shift for estimating weighted mean
 * param[in]: set of prediction votes and initial estimation
 *
 */
void
meanshift(const std::vector<cv::Vec3f>& votes, cv::Vec3f& mean_vote)
{
    // const float bandwidth = pow(0.2, 2.f);
    const float bandwidth = 0.08;
    const int max_ms_iter = 10;

    for (int iter = 0; iter < max_ms_iter; ++iter)
    {
        cv::Vec3f temp_mean = 0;
        float sum_weight = 0.f;

        // compute weighted mean
        for (int k = 0; k < votes.size(); ++k)
        {
            // compute gaussian kernel
            float norm = (float)cv::norm(mean_vote, votes[k]);
            float weight = 1.0 / exp(pow(norm, 2.0) / (2.0 * bandwidth));
            sum_weight += weight;

            // weighted mean
            temp_mean += weight * votes[k];
        }

        // normalize weighted mean
        temp_mean /= (float)MAX(1, sum_weight);

        // compute distance to previous mean
        float distance_to_previous_mean = (float)cv::norm(temp_mean, mean_vote);

        // update mean
        mean_vote = temp_mean;

        // check converge to stop
        if (distance_to_previous_mean < 0.05)
            break;
    }
}


/*
 * program testing
 * param[in]: dataset path and model path
 *
 */
void
testing(const fs::path& dataset_path, const fs::path& model_path)
{
    // load train dataset
    std::vector<rf::sample> data;
    load_dataset(dataset_path, data);

    std::vector<fs::path> tree_paths;
    fs::directory_iterator end_itr;

    // get number of trees in the directory
    for (fs::directory_iterator i(model_path); i != end_itr; ++i)
    {
        // get filename of tree model file
        if (fs::is_regular_file(i->status()) && i->path().extension() == ".yml")
        {
            tree_paths.push_back(i->path());
        }
    }

    // init forest
    std::vector<rf::tree> forest(tree_paths.size());

    // load trees
    for (int i = 0; i < forest.size(); ++i)
    {
        assert(fs::exists(tree_paths[i]));

        rf::tree tree;
        tree.load(tree_paths[i].string());
        forest[i] = std::move(tree);
    }

    cerr << "loaded " << forest.size() << " trees" << endl;

    float mean_error = 0;

    // iterate through test data
    for (int i = 0; i < data.size(); ++i)
    {
        cv::Vec3f mean = 0;
        cv::Vec3f prediction = 0;
        int count = 0;

        // send data to the trees for prediction
        std::vector<cv::Vec3f> votes;
        for (int j = 0; j < forest.size(); ++j)
        {
            rf::leaf_vote vote = forest[j].predict(data[i]);
            votes.push_back(vote.mean);

            mean += vote.mean;
            count++;
        }
        mean /= (float)MAX(1, count);

        assert(votes.size() == forest.size());

        // mean shift to find weighted mean / mode
        meanshift(votes, mean);

        // final prediction
        cout << mean << endl;

        // calculate mean error
        mean_error += (float)cv::norm(data[i].label, mean);
    }

    mean_error /= (float)data.size();
    cout << "-----------------" << endl;
    cout << "mean error : " << mean_error << endl;
}


/*
 * program leave-one-out validation
 * param[in]: dataset path and model path
 *
 */
void
crossval(const fs::path& dataset_path, const fs::path& model_path, const int& num_trees)
{
    const int max_img_idx = 23;

    // load full dataset
    std::vector<rf::sample> data;
    load_dataset(dataset_path, data);

    float mean_cv_error = 0;

    // iterate through image test index
    for (int i = 1; i <= max_img_idx; ++i)
    {
        std::vector<rf::sample> train_set, test_set;

        // collecting data
        // testing image index == i
        // the rest images are used for training
        for (int j = 0; j < data.size(); ++j)
        {
            if (data[j].img_index == i)
            {
                test_set.push_back(data[j]);
            }
            else
            {
                train_set.push_back(data[j]);
            }
        }

        char folder_index[128];
        sprintf(folder_index, "cv_%02d", i);

        // check output folder
        // if not exiss create it
        fs::path out_dir = model_path / folder_index;
        if (!fs::exists(out_dir))
            fs::create_directory(out_dir);

        char tree_index[128];
        std::vector<rf::tree> forest(num_trees);

        // training forest
        for (int t = 0; t < num_trees; ++t)
        {
            // create tree path
            sprintf(tree_index, "tree_%02d.yml", t);
            fs::path tree_path = out_dir / tree_index;

            // sampling with replacement
            std::vector<rf::sample> samples;
            sampling_with_replacement(train_set, samples);

            // construct tree
            cerr << "training " << tree_path << endl;
            rf::tree tree;
            tree.grow(samples);

            // save tree
            tree.save(tree_path.string());
            assert(fs::exists(tree_path));

            // add tree to forest
            forest[t] = std::move(tree);
        }

        float mean_error = 0;

        // testing forest
        for (int j = 0; j < test_set.size(); ++j)
        {
            cv::Vec3f mean = 0;
            cv::Vec3f prediction = 0;
            int count = 0;

            // send data to the trees for prediction
            std::vector<cv::Vec3f> votes;
            for (int j = 0; j < forest.size(); ++j)
            {
                rf::leaf_vote vote = forest[j].predict(test_set[j]);
                votes.push_back(vote.mean);

                mean += vote.mean;
                count++;
            }
            mean /= (float)MAX(1, count);

            // mean shift to find weighted mean / mode
            // mean is the final prediction
            meanshift(votes, mean);

            // calculate error
            mean_error += (float)cv::norm(test_set[j].label, mean);
        }

        // mean error at this evaluation
        mean_error /= (float)test_set.size();
        mean_cv_error += mean_error;
        cout << mean_error << endl;
    }

    mean_cv_error /= (float)max_img_idx;
    cout << "mean cross validation error : " << mean_cv_error << endl;
}


// Main function
int
main(int argc, char *argv[])
{
    fs::path dataset_path, model_path;
    int mode;
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h",
             "display the help message")
            ("data,d", po::value<fs::path>(&dataset_path)->required(),
                "path to dataset")
            ("model,m", po::value<fs::path>(&model_path)->required(),
                "path to model file")
            ("program,p", po::value<int>(&mode)->required(),
                "program mode")
            ;
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help")) {
            cout << "Usage: train [options]" << endl;
            cout << desc;
            return EXIT_SUCCESS;
        }
        po::notify(vm);
    }
    catch (const po::error& e) {
        cout << "Error while parsing command-line arguments: " << e.what() << endl;
        cout << "Use --help to display a list of options." << endl;
        return EXIT_SUCCESS;
    }


    switch (mode)
    {
        case 0:
            clog << "program training" << endl;
            training(dataset_path, model_path);
            break;
        case 1:
            clog << "program testing" << endl;
            testing(dataset_path, model_path);
            break;
        case 2:
            clog << "program leave-one-out evaluation" << endl;
            crossval(dataset_path, model_path, 1);
            break;
        default:
            cout << "error mode input" << endl;;
    }


    return EXIT_SUCCESS;
}
