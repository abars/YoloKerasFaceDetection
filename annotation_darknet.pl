#Generate annotation for yolo

use warnings;
use strict;
use Image::Size;

mkdir "WIDER_train/annotations_darknet";

my $file_no=0;
my $line_no=0;

open(IN,"<wider_face_split/wider_face_train_bbx_gt.txt") or die ("wider face dataset not found");

my $imagew;
my $imageh;

open(TRAIN,">WIDER_train/annotations_darknet/train.txt");
open(TEST,">WIDER_train/annotations_darknet/test.txt");

while(my $line=<IN>){
  #print $line;

  if($line =~ /--/){
    if($line_no ne 0){
      close(OUT);
      $line_no=0;
    }

    $line_no=1;
    $file_no=$file_no+1;
    my $file_path=$line;

    #my $temp_no;
    #my $file_name;
    #($temp_no,$file_name)=split("--",$file_path);
    chomp $file_path;

    ($imagew, $imageh) = imgsize("./WIDER_train/images/$file_path");

    if($file_no%4 eq 0){
      print TEST "../WIDER_train/annotations_darknet/$file_no.jpg\n";
    }else{
      print TRAIN "../WIDER_train/annotations_darknet/$file_no.jpg\n";        
    }

    system("cp ./WIDER_train/images/$file_path WIDER_train/annotations_darknet/$file_no.jpg");
    open(OUT,">WIDER_train/annotations_darknet/$file_no".".txt");
   next;
  }
  if($line_no eq 1){
    $line_no=$line_no+1;
    next;
  }
  if($line_no eq 2){
    my $xmin;
    my $ymin;
    my $w;
    my $h;
    ($xmin,$ymin,$w,$h)=split(" ",$line);
    my $x=$xmin+$w/2;
    my $y=$ymin+$h/2;
    my $category=0;
    $x=1.0*$x/$imagew;
    $y=1.0*$y/$imagew;
    $w=1.0*$w/$imagew;
    $h=1.0*$h/$imagew;
    print OUT "$category $x $y $w $h\n";
  }
}

if($line_no ne 0){
  close(OUT);
}

close(IN);
close(TRAIN);
close(TEST);