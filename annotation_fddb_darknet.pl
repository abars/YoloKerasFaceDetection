#Generate annotation for yolo
#http://vis-www.cs.umass.edu/fddb/

use warnings;
use strict;
use Image::Size;
use File::Copy;

my $dataset_path="dataset/fddb/";

mkdir "$dataset_path"."FDDB-folds/annotations_darknet";

my $file_no=0;

my $imagew;
my $imageh;

open(TRAIN,">$dataset_path"."FDDB-folds/annotations_darknet/train.txt");
open(TEST,">$dataset_path"."FDDB-folds/annotations_darknet/test.txt");

for(my $list=1;$list<=10;$list=$list+1){
  my $list02="$list";
  if($list<10){
    $list02="0$list";
  }
  open(IN,"<$dataset_path"."FDDB-folds/FDDB-fold-".$list02."-ellipseList.txt") or die ("wider face dataset not found");

  while(my $line=<IN>){
    #file name
    #print $line;

    $file_no=$file_no+1;
    my $file_path=$line;

    chomp $file_path;

    ($imagew, $imageh) = imgsize("./$dataset_path"."originalPics/$file_path.jpg");

    if($file_no%4 eq 0){
      print TEST "../$dataset_path"."FDDB-folds/annotations_darknet/$file_no.jpg\n";
    }else{
      print TRAIN "../$dataset_path"."FDDB-folds/annotations_darknet/$file_no.jpg\n";        
    }

    copy("./$dataset_path"."originalPics/$file_path.jpg","$dataset_path"."FDDB-folds/annotations_darknet/$file_no.jpg");
    open(OUT,">$dataset_path"."FDDB-folds/annotations_darknet/$file_no".".txt");

    #line count
    $line=<IN>;
    #print $line;
    my $line_n=int($line);

    for(my $i=0;$i<$line_n;$i=$i+1){
      my $major_axis_radius=0;
      my $minor_axis_radius=0;
      my $angle=0;
      my $center_x=0;
      my $center_y=0;
      $line=<IN>;
      #print $line;
      ($major_axis_radius,$minor_axis_radius,$angle,$center_x,$center_y)=split(" ",$line);

        my $x=$center_x;
        my $y=$center_y;

        #my $w=abs(cos($angle)*$major_axis_radius)*2;
        #my $h=abs(cos($angle)*$minor_axis_radius)*2;

        my $w=$minor_axis_radius*2;
        my $h=$major_axis_radius*2;

        my $category=0;
        $x=1.0*$x/$imagew;
        $y=1.0*$y/$imageh;
        $w=1.0*$w/$imagew;
        $h=1.0*$h/$imageh;

      if($w>0 and $h>0 and $x-$w/2>=0 and $y-$h/2>=0 and $x+$w/2<=1 and $y+$h/2<=1){
        print OUT "$category $x $y $w $h\n";
      }else{
        print "Invalid position removed $x $y $w $h at $file_no\n";
      }
    }

    close(OUT);
  }

  close(IN);
}

close(TRAIN);
close(TEST);