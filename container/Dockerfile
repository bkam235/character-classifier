FROM ubuntu:latest

#RUN useradd nginx

# Copy python requirements file
COPY requirements.txt /tmp/requirements.txt
# COPY ImageMagick.tar.gz /tmp/ImageMagick.tar.gz

RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev build-essential nano
RUN pip3 install --upgrade pip setuptools

RUN pip3 install -r /tmp/requirements.txt
#RUN rm /etc/nginx/conf.d/default.conf
RUN rm -r /root/.cache

# WORKDIR /tmp
# RUN tar xvzf ImageMagick.tar.gz
# WORKDIR /tmp/ImageMagick-7.0.9-27
# RUN ./configure
# RUN make
# RUN sudo make install
# RUN sudo ldconfig /usr/local/lib

RUN apt-get install -y libmagickwand-dev


# Add app
COPY ./app /app
WORKDIR /app

# RUN python3 generate_data.py

# RUN pip install pillow
# RUN pip install pytesseract
# RUN pip install opencv-contrib-python

# CMD ["/usr/bin/supervisord"]

# RUN /bin/bash