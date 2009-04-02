#!/usr/bin/perl -w
use strict;

# input: .p file
# output is printed on terminus. use redirection ">" to save into the desired filename
if(@ARGV !=1) { die "correct usage\n\t: perl make_passive_model.pl <input.p>\n";} 

my $INPUT_FILE = $ARGV[0];

################# main #######################

open(INPUT, $INPUT_FILE) || die "can't open $INPUT_FILE\n";
while (defined(my $line =<INPUT>)){
      my $firstChar = substr($line,0,1);
      if($firstChar eq "\n"){print $line; next;}
      if($firstChar eq "/"){print $line; next;}
      if($firstChar eq "*"){print $line; next;}
      processLine($line);      
}
close(INPUT);

############## internal functions #############

sub processLine{
      my $line = $_[0];
      chomp $line;
      my $keepPrinting =1;
      my @arr = split  /\s/, $line;
      my $count=0; # count keeps the track of which value we are reading in the current line
      for(my $i=0; $i< @arr; $i++){
            if($arr[$i] eq ""){print "\t"; next;}
            $count++;
            if($count > 2 and !isNumeric($arr[$i])){last;}            
            else { print "$arr[$i]\t";}
      }                        
      print "\n";
}


# return 1 if the input is a number
# this is not full-proof, but OK for our purpose (the coordinates in the .p files will not have any characters in them (including "e").. 
sub isNumeric{
	my $value = $_[0];
	if($value =~ /[a-zA-Z]/){ return 0;}
	return 1;
}
