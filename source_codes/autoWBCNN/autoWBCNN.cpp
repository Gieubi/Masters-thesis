#include <iostream>
#include <memory>
#include <stdio.h>
// OPENCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
// TORCH
#include <torch/script.h>
#include <torch/torch.h>
using namespace torch; using namespace torch::nn;

auto ToInput(at::Tensor tensor_image)
{
    // Create a vector of inputs.
    return std::vector<torch::jit::IValue>{tensor_image};
}

int main(int argc, const char* argv[]) 
{
    //Get image name, path and load it
    std::string imagePath = argv[1];
    cv::Mat img = cv::imread(imagePath);

    // downsizing images for better performance
    int new_height = img.size().height - img.size().height%16;
    int new_width = img.size().width - img.size().width%16;
    
    new_height /= 4;
    new_width /= 4;
    cv::Mat resized_down;

    // resize down
    resize(img, resized_down, cv::Size(new_width, new_height), cv::INTER_LINEAR);
    img = resized_down;

    // convert the image into float type and scale it (model has float weights)
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);

    // create a tensor with model from the image
    auto tensor = torch::from_blob(img.data, {1, img.size().height, img.size().width, 3});

    // change dimensions compatibly with the model
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = tensor.to(torch::kCPU);

    // transform the tensor into a input vector
    std::vector<torch::jit::IValue> input_to_net;
    input_to_net.push_back(tensor);
    try
    {
        std::string AWB_model_path = "/path/to/traced_AWB_CPU.zip";
        
        // Deserialize the ScriptModule from a file using torch::jit::load().
        torch::jit::script::Module AWB = torch::jit::load(AWB_model_path);
        AWB.eval();
          
    // Execute the model and turn its output into a tensor.
    at::Tensor out_tensor = AWB.forward(input_to_net).toTensor();
    // convert the tensor back into image
    out_tensor = out_tensor.squeeze().detach().permute({1, 2, 0});
    out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kCPU);
    out_tensor = out_tensor.to(torch::kU8);
    cv::Mat resultImg(img.size().height, img.size().width, CV_8UC3, out_tensor.data_ptr());

    // save the result
    cv::imwrite("cpp_AWB_" + imagePath, resultImg);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "error loading the model\n" <<e.msg();
        return -1;
    }  
    return 0;
}