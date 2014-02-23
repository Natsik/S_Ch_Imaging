S_Ch_Imaging
------------

Тестовые примеры свзяки C + Python с использованием Cython
Необходимые библиотеки: Numpy (для numpy_cos, multiply), Cython
build:

    python setup.py build_ext --inplace --compiler=mingw32

    >>> import helloworld
    Hello, world!

После билда должны получаться файлы *.pyd
Все импорты не должны кидать ошибок.