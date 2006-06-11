//genesis

echo doing Fibonacci series from 1 to 20

int i
int a = 0
int b = 1
int c

echo 1
for (i = 0 ; i < 20; i = i + 1)
	c = a + b
	echo {c}
	a = b
	b = c
end
