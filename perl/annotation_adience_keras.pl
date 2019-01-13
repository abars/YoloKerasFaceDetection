#Generate annotation for keras
#https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

use warnings;
use strict;
use Image::Size;
use File::Copy;

my $input_path="dataset/adience/";
my $dataset_path="dataset/agegender_adience/";

my $ANNOTATION_FILES="./$input_path"."fold_";
my $FACE_FILES="./$input_path"."aligned";

mkdir "$dataset_path"."annotations/agegender/";
mkdir "$dataset_path"."annotations/agegender/validation";
mkdir "$dataset_path"."annotations/agegender/train";

mkdir "$dataset_path"."annotations/gender";
mkdir "$dataset_path"."annotations/gender";
mkdir "$dataset_path"."annotations/gender/validation";
mkdir "$dataset_path"."annotations/gender/train";

mkdir "$dataset_path"."annotations/age";
mkdir "$dataset_path"."annotations/age";
mkdir "$dataset_path"."annotations/age/validation";
mkdir "$dataset_path"."annotations/age/train";

my $file_no=0;
my $line_no=0;

my $imagew;
my $imageh;

my $before_file="";

for(my $list=0;$list<5;$list=$list+1){
  open(IN,"$ANNOTATION_FILES"."$list"."_data.txt") or die ("agegender dataset not found");

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
    my $category_label="";
    my $age_int=-1;
    if($age =~ /^[0-9]+$/){
      $age_int=int($age);
    }else{
      if($age =~ /\(([0-9]+),/){
        $age_int=int("$1");
      }
    }

    if($age_int>=0 and $age_int<=3){
      $category=0;
      $category_label="0-2";
    }
    if($age_int>=4 && $age_int<=7){
      $category=1;
      $category_label="4-6";
    }
    if($age_int>=8 && $age_int<=14){
      $category=2;
      $category_label="8-13";
    }
    if($age_int>=15 && $age_int<=24){
      $category=3;
      $category_label="15-20";
    }
    if($age_int>=25 && $age_int<=37){
      $category=4;
      $category_label="25-32";
    }
    if($age_int>=38 && $age_int<=47){
      $category=5;
      $category_label="38-43";
    }
    if($age_int>=48 && $age_int<=59){
      $category=6;
      $category_label="48-53";
    }
    if($age_int>=60 && $age_int<=100){
      $category=7;
      $category_label="60-";
    }
    if($age eq "None"){
      #skip unknown age from dataset
      next;
    }
    if($category eq -1){
      print "Age not found $age\n$line\n";
      exit;
    }

    if($gender eq "f"){
      $category=$category+8;
    }else{
      if($gender eq "m"){
        $category=$category+0;
      }else{
        #print "Gender not found $gender\n$line\n";
        next;
      }
    }

    my $label="$category"."_"."$category_label"."_"."$gender";
    my $label_age="$category_label";
    my $label_gender="$gender";

    my $thumb_dir="$FACE_FILES/$user_id/";

    opendir(THUMB, "$thumb_dir") or die "usage: $0 thumb_dir\n";
    my $filepath="";
    foreach my $dir (readdir(THUMB)) {
      next if ($dir eq '.' || $dir eq '..');
      next if ($dir eq '.DS_Store');
      if($dir =~ /$face_id\.$original_image/){
        $filepath=$dir;
        last;
      }
    }

    if($filepath eq ""){
      print "image file not found\n";
      next;
    }

    mkdir "./$dataset_path"."annotations/agegender/validation/$label";
    mkdir "./$dataset_path"."annotations/agegender/train/$label";

    mkdir "./$dataset_path"."annotations/age/validation/$label_age";
    mkdir "./$dataset_path"."annotations/age/train/$label_age";

    mkdir "./$dataset_path"."annotations/gender/validation/$label_gender";
    mkdir "./$dataset_path"."annotations/gender/train/$label_gender";

    if($before_file ne $original_image){
      #if($line_no ne 0){
      #  close(OUT);
      #  $line_no=0;
      #}
      $before_file=$original_image;
      #$line_no=1;
      ($imagew, $imageh) = imgsize("$FACE_FILES/$user_id/$filepath");

      $file_no=$file_no+1;
      if($file_no%4 eq 0){
        copy("$FACE_FILES/$user_id/$filepath","$dataset_path"."annotations/agegender/validation/$label/$filepath");
        copy("$FACE_FILES/$user_id/$filepath","$dataset_path"."annotations/age/validation/$label_age/$filepath");
        copy("$FACE_FILES/$user_id/$filepath","$dataset_path"."annotations/gender/validation/$label_gender/$filepath");
      }else{
        copy("$FACE_FILES/$user_id/$filepath","$dataset_path"."annotations/agegender/train/$label/$filepath");
        copy("$FACE_FILES/$user_id/$filepath","$dataset_path"."annotations/age/train/$label_age/$filepath");
        copy("$FACE_FILES/$user_id/$filepath","$dataset_path"."annotations/gender/train/$label_gender/$filepath");
      }
    }
    
    $x=1.0*$x/$imagew;
    $y=1.0*$y/$imagew;
    $w=1.0*$w/$imagew;
    $h=1.0*$h/$imagew;
    #print OUT "$category $x $y $w $h\n";
  }

  #if($line_no ne 0){
  #  close(OUT);
  #}

  close(IN);
}
