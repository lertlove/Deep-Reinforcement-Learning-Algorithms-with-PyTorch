#Install a prerequisite packages which let apt utilize HTTPS:
sudo apt -y install apt-transport-https ca-certificates curl software-properties-common

#Add GPG key for the official Docker repo to the Ubuntu system:
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

#Add the Docker repo to APT sources:
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"

#Update the  database with the Docker packages from the added repo:
sudo apt -y update

#Install Docker software:
sudo apt -y install docker-ce

#Docker should now be installed, the daemon started, and the process enabled to start on boot. To verify:
# sudo systemctl status docker
 
#NOTE: To avoid using sudo for docker activities, add your username to the Docker Group
sudo usermod -aG docker ${USER}
newgrp docker

#Docker Compose
#The command below will download the 1.28.5 release and save the executable at /usr/local/bin/docker-compose, which will make this software globally accessible as docker-compose:
sudo curl -L "https://github.com/docker/compose/releases/download/1.28.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

#Set permissions so that the docker-compose command is executable:
sudo chmod +x /usr/local/bin/docker-compose

#Verify that the installation was successful by viewing version information:
docker-compose --version

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get -y update
sudo apt-get -y install nvidia-docker2
sudo systemctl restart docker.service
