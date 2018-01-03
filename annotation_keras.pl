#Generate annotation

use warnings;
use strict;
use Image::Size;

mkdir "WIDER_train/annotations_keras";

my $file_no=0;
my $line_no=0;

open(IN,"<wider_face_split/wider_face_train_bbx_gt.txt") or die ("wider face dataset not found");

while(my $line=<IN>){
  #print $line;

  if($line =~ /--/){
    if($line_no ne 0){
      print OUT <<"EOF";
</annotation>
EOF
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

    my $imagew;
    my $imageh;
    ($imagew, $imageh) = imgsize("./WIDER_train/images/$file_path");

    open(OUT,">WIDER_train/annotations_keras/wider_face-$file_no".".xml");
    print OUT <<"EOF";
<annotation verified="yes">
  <folder>images</folder>
  <filename>$file_path</filename>
  <path>$file_path</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>$imagew</width>
    <height>$imageh</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
EOF
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
    my $xmax;
    my $ymax;
    $xmax=$xmin+$w;
    $ymax=$ymin+$h;
    print OUT <<"EOF";
  <object>
    <name>face</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>$xmin</xmin>
      <ymin>$ymin</ymin>
      <xmax>$xmax</xmax>
      <ymax>$ymax</ymax>
    </bndbox>
  </object>
EOF
  }
}

if($line_no ne 0){
  print OUT <<"EOF";
</annotation>
EOF
  close(OUT);
}

close(IN);