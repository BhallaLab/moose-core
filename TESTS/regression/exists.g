create neutral n
int bool = {exists n}
if (bool == 1)
	echo "Pass: No problems. Detects /n"
else 
	echo "Error: Not detecting /n"
end

bool = {exists n x} 
if (bool == 1 )
	echo "Error: Wrongly detecting x field in /n"
else 
	echo "Pass: x field is not in /n"
end

addfield n x

bool = {exists n x}
if (bool == 1)
        echo "Pass: Detects x field in /n"
else 
	echo "Error: Not detecting x field in /n"
end

