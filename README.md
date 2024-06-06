# EDGEbin
![edgeBinlogo](https://github.com/AnjananT/EDGEbin/assets/144964837/60c39b1f-60da-4e51-b562-c7d3e73dce7f)

# What it does
The EDGEbin utilizes artificial intelligence and a clever mechanical system to automatically identify and sort your trash into four categories: trash, recycling, electronic, and organic. All you have to do is toss your trash into the funnel. Afterwards, the EDGEbin website allows you to track how full your bin is and update when it has been emptied.

# How it was built
AI: Employing PyTorch and a custom dataset, I trained, validated, and tested a custom neural network for waste classification. For redundancy, a second neural network was made with a pretrained model finetuned to our classifications. These models, coupled with OpenCV, identified and classified waste via a webcam. Then, using the PySerial library, this information was sent to the Arduino. 

Hardware: The physical system, designed in SolidWorks, came to life through the use of laser cutting and 3D printing. Using C++, the Arduino was programmed to control a DC motor that moves an inner tube. Using the Stepper motor library and a motor driver, the stepper motor controlled the opening and closing of the release hatch. These electrical components were meticulously wired and seamlessly integrated with their physical counterparts. 

Website: The frontend was build using Flask to handle the backend, and ReactJS for the frontend.
