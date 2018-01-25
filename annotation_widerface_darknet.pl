#Generate annotation for yolo
#http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

use warnings;
use strict;
use Image::Size;
use File::Copy;

my $dataset_path="dataset/widerface/";

mkdir "$dataset_path"."WIDER_train/annotations_darknet";

my $file_no=0;
my $line_no=0;

open(IN,"<$dataset_path"."wider_face_split/wider_face_train_bbx_gt.txt") or die ("wider face dataset not found");

my $imagew;
my $imageh;

open(TRAIN,">$dataset_path"."WIDER_train/annotations_darknet/train.txt");
open(TEST,">$dataset_path"."WIDER_train/annotations_darknet/test.txt");

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

    chomp $file_path;

    ($imagew, $imageh) = imgsize("./$dataset_path"."WIDER_train/images/$file_path");

    if($file_no%4 eq 0){
      print TEST "../$dataset_path"."WIDER_train/annotations_darknet/$file_no.jpg\n";
    }else{
      print TRAIN "../$dataset_path"."WIDER_train/annotations_darknet/$file_no.jpg\n";        
    }

    copy("./$dataset_path"."WIDER_train/images/$file_path","$dataset_path"."WIDER_train/annotations_darknet/$file_no.jpg");
    open(OUT,">$dataset_path"."WIDER_train/annotations_darknet/$file_no".".txt");
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

    if($w>0 and $h>0 and $xmin>=0 and $ymin>=0 and $xmin+$w<=$imagew and $ymin+$h<=$imageh){
      my $x=$xmin+$w/2;
      my $y=$ymin+$h/2;
      my $category=0;
      $x=1.0*$x/$imagew;
      $y=1.0*$y/$imageh;
      $w=1.0*$w/$imagew;
      $h=1.0*$h/$imageh;
      print OUT "$category $x $y $w $h\n";
    }else{
      print "Invalid position removed $xmin $ymin $w $h at $file_no\n";
    }
  }
}

if($line_no ne 0){
  close(OUT);
}

close(IN);
close(TRAIN);
close(TEST);