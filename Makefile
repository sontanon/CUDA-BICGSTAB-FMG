NVCC = nvcc
NVCCFLAGS = -arch=sm_50 -O2 -lcublas -lcusolver -lcusparse
NVCCFLAGS2 = -dc -I.

all: twoStep.exe

twoStep.exe: main.obj tools.obj csrGenerator.obj
	$(NVCC) $(NVCCFLAGS) main.obj tools.obj csrGenerator.obj -o twoStep.exe

main.obj: main.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS2) main.cu -o main.obj

tools.obj: tools.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS2) tools.cu -o tools.obj

csrGenerator.obj: csrGenerator.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS2) csrGenerator.cu -o csrGenerator.obj

clean:
	del *.obj twoStep.exp twoStep.lib twoStep.exe
