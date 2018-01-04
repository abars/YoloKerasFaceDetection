#Generate annotation for yolo
#https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

use warnings;
use strict;
use Image::Size;
use File::Copy;

mkdir "agegender/annotations";

my $file_no=0;
my $line_no=0;

my $imagew;
my $imageh;

my $before_file="";

open(TRAIN,">agegender/annotations/train.txt");
open(TEST,">agegender/annotations/test.txt");

for(my $list=0;$list<5;$list=$list+1){
  open(IN,"<agegender/fold_"."$list"."_data.txt") or die ("agegender dataset not found");

  my $header=<IN>;

  while(my $line=<IN>){
    #print $line;

    my $user_id;
    my $original_image;
    my $face_id;
    my $age;
    my $gender;
    my $x;
    my $y;
    my $dx;
    my $dy;
    my $tilt_ang;
    my $fiducial_yaw_angle;
    my $fiducial_score;

    ($user_id,$original_image,$face_id,$age,$gender,$x,$y,$dx,$dy,$tilt_ang,$fiducial_yaw_angle,$fiducial_score)=split("\t",$line);

    $x=$x+$dx/2;
    $y=$y+$dy/2;
    my $w=$dx;
    my $h=$dy;
    
    my $category=-1;
    if($age eq "(0, 2)"){
      $category=0;
    }
    if($age eq "(4, 6)"){
      $category=1;
    }
    if($age eq "(8, 13)"){
      $category=2;
    }
    if($age eq "(15, 20)"){
      $category=3;
    }
    if($age eq "(25, 32)"){
      $category=4;
    }
    if($age eq "(38, 43)"){
      $category=5;
    }
    if($age eq "(48, 53)"){
      $category=6;
    }
    if($age eq "(60, 100)"){
      $category=7;
    }
    if($age eq "None"){
      next;
    }
    if($category eq -1){
      print "Age not found $age";
      exit;
    }

    if($gender eq "f"){
      $category=$category+8;
    }

    if($before_file ne $original_image){
      if($line_no ne 0){
        close(OUT);
        $line_no=0;
      }
      $before_file=$original_image;
      $line_no=1;
      ($imagew, $imageh) = imgsize("./agegender/images/$original_image");

      $file_no=$file_no+1;
      if($file_no%4 eq 0){
        print TEST "../agegender/annotations/$original_image\n";
      }else{
        print TRAIN "../agegender/annotations/$original_image\n";        
      }

      copy("./agegender/images/$original_image","agegender/annotations/$file_no.jpg");
      open(OUT,">agegender/annotations/$file_no".".txt");
    }
    
    $x=1.0*$x/$imagew;
    $y=1.0*$y/$imagew;
    $w=1.0*$w/$imagew;
    $h=1.0*$h/$imagew;
    print OUT "$category $x $y $w $h\n";
  }

  if($line_no ne 0){
    close(OUT);
  }

  close(IN);
}
close(TRAIN);
close(TEST);