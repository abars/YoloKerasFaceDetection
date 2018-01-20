#Generate annotation for yolo
#http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/

use warnings;
use strict;
use Image::Size;
use File::Copy;
use File::Spec::Functions qw/catfile/;

my $single_class=1;

for(my $test=0;$test<1;$test=$test+1){ #set 1 because not exist test/posGt
  my $folder="train";
  if($test eq 1){
    $folder="test";
  }

  my $file_no=0;
  my $line_no=0;

  my $thumb_dir="./detectiondata/$folder/posGt";

  open(TRAIN,">detectiondata/$folder/pos/$folder.txt");

  opendir(THUMB, $thumb_dir) or die "usage: $0 thumb_dir\n";
  foreach my $dir (readdir(THUMB)) {
    next if ($dir eq '.' || $dir eq '..');
    next if ($dir eq '.DS_Store');
    my $tag_dir = catfile($thumb_dir, $dir);

    #print $dir;

    open(IN,"<$tag_dir") or die ("wider face dataset not found");

    my $imagew;
    my $imageh;

    my $header=<IN>;

    my $file_path=$dir;

    chomp $file_path;
    #print $file_path."\n";

    my $img_path=$file_path;
    $img_path =~ s/\.txt/\.png/g;
    #print $img_path."\n";
    ($imagew, $imageh) = imgsize("./detectiondata/$folder/pos/$img_path");

    print TRAIN "../detectiondata/$folder/pos/$img_path\n";

    open(OUT,">./detectiondata/$folder/pos/$file_path");

    while(my $line=<IN>){
      #print $line;
      my $category_name;
      my $xmin;
      my $ymin;
      my $w;
      my $h;
      ($category_name,$xmin,$ymin,$w,$h)=split(" ",$line);
      my $x=$xmin+$w/2;
      my $y=$ymin+$h/2;
      $x=1.0*$x/$imagew;
      $y=1.0*$y/$imageh;
      $w=1.0*$w/$imagew;
      $h=1.0*$h/$imageh;

      my $category=0;
      if($category_name eq "leftHand_driver"){
        $category=0;
      }else{
        if($category_name eq "rightHand_driver"){
          $category=1;
        }else{
          if($category_name eq "leftHand_passenger"){
            $category=2;
          }else{
            if($category_name eq "rightHand_passenger"){
                $category=3;
              }else{
                print "unknown $category_name";
              }
          }
        }
      }

      if($single_class eq 1){
        $category=0;
      }

      print OUT "$category $x $y $w $h\n";
    }

    close(OUT);
    close(IN);
  }
  close(THUMB);

  close(TRAIN);
}
