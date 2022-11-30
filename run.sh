sudo docker build -t pf .
sudo docker rm myapp
sudo docker run --name myapp --rm --net=host -p 8000:8000  pf:latest
