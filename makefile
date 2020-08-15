clpath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.26.28801\bin"

default: lux.cu ray.h resources.res resource.h
	nvcc -o lux -include resource.h user32.lib lux.cu resources.res -ccbin $(clpath)

resources.res: resources.rc resource.h
	rc -nologo -fo resources.res resources.rc

clean:
	del *.res *.png lux.lib lux.exp lux.exe *.txt