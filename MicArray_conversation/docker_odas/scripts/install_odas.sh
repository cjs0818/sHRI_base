apt-get install -y libfftw3-dev libconfig-dev libasound2-dev libpulse-dev
apt-get install -y usbutils

cd /root/work/
git clone https://github.com/introlab/odas.git
cd odas
mkdir build
cd build
cmake ..
make