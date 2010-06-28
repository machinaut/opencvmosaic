PACKAGES = opencv
CPPFLAGS := $(shell pkg-config --cflags $(PACKAGES))  -Wall
LDFLAGS := $(shell pkg-config --libs $(PACKAGES))

OBJS = opencvmosaic.o
CC = g++

TARGET = opencvmosaic

all: $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^  -o $@
.o:
	$(CC) $(CPPFLAGS) $@.c -o $@

