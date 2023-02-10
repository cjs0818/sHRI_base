# 1. Initial Setting

## 1.1 Install Anaconda on the NUC PC

* Visit the Anaconda Distribution page.
  * https://www.anaconda.com/products/distribution


* Download the recommended version of Anaconda for Linux.
  * Ubuntu 22.04 was used for existing projects.


* Run the downloaded Install File.
  ```
  bash "downloaded Install File name".sh
  ```

* Press Enter, answer 'yes', press Enter again to confirm the location.

* answer 'yes' to initialize Anaconda3 by running conda init.

* Update the bashrc
  ```
  source ~/.bashrc
  ```

* The installation is complete if (base) appears before the command.    
</br>

## 1.2. Install pip on the Linux.

* Update system repository.
  ```
  sudo apt update
  sudo apt upgrade -y
  ```

* Install Python3 pip.
  ```
  sudo apt install python3-pip -y
  ```

* Confirm the Pip Version and Installation
  ```
  pip3 --version
  ```
</br>

## 1.3. Install git on the Linux.

* Update system repository.
  ```
  sudo apt update
  sudo apt upgrade -y
  ```

* Install git.
  ```
  sudo apt install git -y
  ```

* Confirm the Git Version and Installation.
  ```
  git --version
  ```
</br>

## 1.4. Make or Clone "HRI" folder in Home.

* All modules worked and ran from HRI folder within Home path.
  ```
  git clone https://github.com/WoongDemianPark/HRI.git
  cd HRI
  ```

</br>

# 2. MicArray Module Setting

[Follow the MicArray Module setting guide.](https://github.com/WoongDemianPark/HRI/tree/main/MicArray)

</br>

# 3. Speaker Diarization (diart) Module Setting

[Follow the Speaker Diarazation (diart) Module setting guide.](https://github.com/WoongDemianPark/HRI/tree/main/diart)

</br>

# 4. Face & Emotion Recognition Module Setting

[Follow the Realtime Face Detection and Emotion Recognition Module setting guide.](https://github.com/WoongDemianPark/HRI/tree/main/FaceDetEmo)

</br>

# 5. Robot Facial Expression Module Setting

[Follow the Robot Facial Expression Module setting guide.](https://github.com/WoongDemianPark/HRI/tree/main/FacialExpression)

</br>

# 6. Combined Robot Facial Module Setting

[Follow Face Detection/Recognition/Expression Module setting guide.](https://github.com/WoongDemianPark/HRI/tree/main/FacialDetExp)

</br>

# 7. Speech to Text Module Setting

[Follow the Speech to Text Module setting guide.](https://github.com/WoongDemianPark/HRI/tree/main/STT)

</br>

