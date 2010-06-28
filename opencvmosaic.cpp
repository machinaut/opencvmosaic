// openCV / SURF based mosaicking  
// Matt Hazard
// NCSU Aerial Robotics Club
// 10 September 2009
// mwhazard@ncsu.edu
// - based on OpenCV find_obj example by Liu Liu (see below)

/*
 * A Demo to OpenCV Implementation of SURF
 * Further Information Refer to "SURF: Speed-Up Robust Feature"
 * Author: Liu Liu
 * liuliu.1987+opencv@gmail.com
 */

#include <cv.h>
#include <highgui.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>

using namespace std;

IplImage *image = 0;

struct mosaic_result
{
	//TODO: H needs to be allocated with the correct dimensions
	CvMat * H; //The homography (mapping matrix) defining the relationship between the two input images
};

void printHomography(CvMat * H)
{
	//cout<<"Recovered homography matrix: " <<endl;
	for (int j = 0; j < 3; j++)
	{
		cout<<" | ";
		for (int k = 0 ; k < 3; k++)
		{
			cout<< H -> data.db[j * 3 + k]<<'\t';
		}
		cout<<" |"<<endl;
	}
}


double
compareSURFDescriptors( const float* d1, const float* d2, double best, int length )
{
    double total_cost = 0;
    assert( length % 4 == 0 );
    for( int i = 0; i < length; i += 4 )
    {
        double t0 = d1[i] - d2[i];
        double t1 = d1[i+1] - d2[i+1];
        double t2 = d1[i+2] - d2[i+2];
        double t3 = d1[i+3] - d2[i+3];
        total_cost += t0*t0 + t1*t1 + t2*t2 + t3*t3;
        if( total_cost > best )
            break;
    }
    return total_cost;
}

int
naiveNearestNeighbor( const float* vec, int laplacian,
                      const CvSeq* model_keypoints,
                      const CvSeq* model_descriptors )
{
    int length = (int)(model_descriptors->elem_size/sizeof(float));
    int i, neighbor = -1;
    double d, dist1 = 1e6, dist2 = 1e6;
    CvSeqReader reader, kreader;
    cvStartReadSeq( model_keypoints, &kreader, 0 );
    cvStartReadSeq( model_descriptors, &reader, 0 );

    for( i = 0; i < model_descriptors->total; i++ )
    {
        const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
        const float* mvec = (const float*)reader.ptr;
        CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        if( laplacian != kp->laplacian )
            continue;
        d = compareSURFDescriptors( vec, mvec, dist2, length );
        if( d < dist1 )
        {
            dist2 = dist1;
            dist1 = d;
            neighbor = i;
        }
        else if ( d < dist2 )
            dist2 = d;
    }
    if ( dist1 < 0.6*dist2 )
        return neighbor;
    return -1;
}

void
findPairs( const CvSeq* objectKeypoints, const CvSeq* objectDescriptors,
           const CvSeq* imageKeypoints, const CvSeq* imageDescriptors, vector<int>& ptpairs )
{
    int i;
    CvSeqReader reader, kreader;
    cvStartReadSeq( objectKeypoints, &kreader );
    cvStartReadSeq( objectDescriptors, &reader );
    ptpairs.clear();

    for( i = 0; i < objectDescriptors->total; i++ )
    {
        const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
        const float* descriptor = (const float*)reader.ptr;
        CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        int nearest_neighbor = naiveNearestNeighbor( descriptor, kp->laplacian, imageKeypoints, imageDescriptors );
        if( nearest_neighbor >= 0 )
        {
            ptpairs.push_back(i);
            ptpairs.push_back(nearest_neighbor);
        }
    }
}

/* a rough implementation for object location */
int
locatePlanarObject( const CvSeq* objectKeypoints, const CvSeq* objectDescriptors,
                    const CvSeq* imageKeypoints, const CvSeq* imageDescriptors,
                    const CvPoint src_corners[4], CvPoint dst_corners[4], CvMat * homography )
{
	if(!homography)
	{
		return 0; //This cannot be NULL!
	}


    vector<int> ptpairs;
    vector<CvPoint2D32f> pt1, pt2;
    CvMat _pt1, _pt2;
    int i, n;

    findPairs( objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors, ptpairs );
    n = ptpairs.size()/2;
    if( n < 4 )
        return 0;

    pt1.resize(n);
    pt2.resize(n);
    for( i = 0; i < n; i++ )
    {
        pt1[i] = ((CvSURFPoint*)cvGetSeqElem(objectKeypoints,ptpairs[i*2]))->pt;
        pt2[i] = ((CvSURFPoint*)cvGetSeqElem(imageKeypoints,ptpairs[i*2+1]))->pt;
    }

    _pt1 = cvMat(1, n, CV_32FC2, &pt1[0] );
    _pt2 = cvMat(1, n, CV_32FC2, &pt2[0] );
    if( !cvFindHomography( &_pt1, &_pt2, homography, CV_RANSAC, 2 ))
        return 0;

    for( i = 0; i < 4; i++ )
    {
        double x = src_corners[i].x, y = src_corners[i].y;
	    double Z = 1./(homography->data.db[6]*x + homography->data.db[7]*y + homography->data.db[8]);
        double X = (homography->data.db[0]*x + homography->data.db[1]*y + homography->data.db[2])*Z;
        double Y = (homography->data.db[3]*x + homography->data.db[4]*y + homography->data.db[5])*Z;
        dst_corners[i] = cvPoint(cvRound(X), cvRound(Y));
    }
    return 1;
}

int main(int argc, char** argv)
{
  if (argc < 4) {
    printf("Usage:\n\topencvmosaic <input1> <input2> <output>\n");
    exit(1);
  }

	//Load filename arguments or use defaults
    const char* object_filename = argc >= 3 ? argv[1] : argv[1];
    const char* scene_filename = argc >= 3 ? argv[2] : argv[2];
	const char * output_filename;
	if(argc >= 4)
	{
		 output_filename = argv[3];
	}
	else
		 output_filename = argv[3];

	int downsample = 1;
	if(argc >=5)
	{
		//disable downsampling
		downsample = 0;
	}

	//TODO: add argument error checking.

    CvMemStorage* storage = cvCreateMemStorage(0);

    //cvNamedWindow("Object", 1);
    //cvNamedWindow("Object Correspond", 1);

	//Define some colors to use when we draw into an image structure
    static CvScalar colors[] = 
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}},
        {{255,255,255}}
    };

	//Load the images, converting to grayscale (TODO: would it help to use color?)
    IplImage* raw_object = cvLoadImage( object_filename, CV_LOAD_IMAGE_GRAYSCALE );
    IplImage* raw_image = cvLoadImage( scene_filename, CV_LOAD_IMAGE_GRAYSCALE );
	IplImage * color_object = cvLoadImage(object_filename, CV_LOAD_IMAGE_COLOR);
	IplImage * color_image = cvLoadImage(scene_filename, CV_LOAD_IMAGE_COLOR);
	//IplImage* object = cvLoadImage( object_filename, CV_LOAD_IMAGE_COLOR );
	//IplImage* image = cvLoadImage( scene_filename, CV_LOAD_IMAGE_COLOR );
	//catch nulls due to image loading failure
    if( !raw_object || !raw_image )
    {
        fprintf( stderr, "Can not load %s and/or %s\n"
            "Usage: find_obj [<object_filename> <scene_filename>]\n",
            object_filename, scene_filename );
        exit(-1);
    }


	//downsample the input images to get faster execution time and possibly better matching...
	CvSize objectDownSize = cvGetSize(raw_object);
	CvSize imageDownSize = cvGetSize(raw_image);

	if(downsample)
	{
	objectDownSize.width /= 2;
	objectDownSize.height /=2;
	imageDownSize.width /= 2;
	imageDownSize.height /=2;
	}
	
		
	IplImage* object = cvCreateImage(objectDownSize, raw_object->depth, raw_object->nChannels);
	IplImage* image = cvCreateImage(imageDownSize, raw_image->depth, raw_image->nChannels);

	cvResize(raw_object, object, CV_INTER_AREA);
	cvResize(raw_image, image, CV_INTER_AREA);



	
    IplImage* object_color = cvCreateImage(cvGetSize(object), 8, 3);
    cvCvtColor( object, object_color, CV_GRAY2BGR );
	//we don't need to convert to color if we load color images to start with (or is this just for drawing?)
    
    CvSeq *objectKeypoints = 0, *objectDescriptors = 0;
    CvSeq *imageKeypoints = 0, *imageDescriptors = 0;
    int i;

    CvSURFParams params = cvSURFParams(500, 1);

    double t1 = (double)cvGetTickCount();
    cvExtractSURF( object, 0, &objectKeypoints, &objectDescriptors, storage, params );
    printf("Object Descriptors: %d\n", objectDescriptors->total);
    cvExtractSURF( image, 0, &imageKeypoints, &imageDescriptors, storage, params );
    printf("Image Descriptors: %d\n", imageDescriptors->total);
    double t2 = (double)cvGetTickCount();
	printf( "Extraction time = %gms\n", (t2-t1) /(cvGetTickFrequency()*1000.));
    CvPoint src_corners[4] = {{0,0}, {object->width,0}, {object->width, object->height}, {0, object->height}};
    CvPoint dst_corners[4];
    
#if 0
	IplImage* correspond = cvCreateImage( cvSize(image->width, object->height+image->height), 8, 1 );
    cvSetImageROI( correspond, cvRect( 0, 0, object->width, object->height ) );
    cvCopy( object, correspond );
    cvSetImageROI( correspond, cvRect( 0, object->height, correspond->width, correspond->height ) );
    cvCopy( image, correspond );
    cvResetImageROI( correspond );
#endif
	//Try to find the homography that maps the planar reference object into the planar scene
	CvMat * H = cvCreateMat(3, 3, CV_64F);
    if( locatePlanarObject( objectKeypoints, objectDescriptors, imageKeypoints,
        imageDescriptors, src_corners, dst_corners, H ))
    {
		printHomography(H);
        for( i = 0; i < 4; i++ )
        {
            CvPoint r1 = dst_corners[i%4];
            CvPoint r2 = dst_corners[(i+1)%4];
            //cvLine( correspond, cvPoint(r1.x, r1.y+object->height ), cvPoint(r2.x, r2.y+object->height ), colors[8] );
        }
    }

	else
	{
		std::cout<<"registration match failed hard - are you sure these images overlap?"<<endl;
		return 1;
	}

	//Compute the minimum bounding rectangle that could contain both images
	CvPoint img_corners[4] = {{0,0}, {image->width,0}, {image->width, image->height}, {0, image->height}};
	CvSeq * cornerSequence = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	
	//cvSeqPushMulti(cornerSequence, img_corners, 4, CV_BACK);
	cvSeqPushMulti(cornerSequence, dst_corners, 4, CV_BACK);
	CvRect mbr = cvBoundingRect(cornerSequence, 0);
	
	cout<<"minimum bounding rectangle: {x: "<<mbr.x<<"  y: "<<mbr.y<<"  width: "<<mbr.width<<"  height: "<<mbr.height<<endl;

	CvMat * H_shift = cvCreateMat(3,3, CV_64F);
	CvMat * H_shift2 = cvCreateMat(3,3, CV_64F);
	CvMat * H_combo = cvCreateMat(3,3, CV_64F);
	CvMat * H_combo2 = cvCreateMat(3,3, CV_64F);
	CvMat * H_scale = cvCreateMat(3,3, CV_64F);
	
	cvSetIdentity(H_shift);
	cvSetIdentity(H_shift2);
	cvSetIdentity(H_scale);
	H_scale->data.db[0] = 0.5;
	H_scale->data.db[4] = 0.5;
	H_scale->data.db[8] = 1;

	int x_offset = mbr.x;
	int y_offset = mbr.y;
	//if(mbr.x < 0)
		H_shift->data.db[2] = -mbr.x;
		H_shift2->data.db[2] = -(mbr.x * 2) + mbr.width/2; 
	//if(mbr.y < 0)
		H_shift->data.db[5] = -mbr.y;
		H_shift2->data.db[5] = -(mbr.y * 2) + mbr.height/2;
	//concatenate the transformations to shift the image into positive coordinates
	cout<<endl<<"shift transformation: "<<endl;
	printHomography(H_shift);
	//cvGEMM(H_shift, H,1, 0, 0, H_combo, 0);
	cvMatMul(H_shift, H, H_combo);
	cvMatMul(H_shift2, H, H_combo2);
	cvMatMul(H_combo2, H_scale, H_scale);

	

	cout<<endl<<"combined transformation: "<<endl;
	printHomography(H_combo);

	for( i = 0; i < 4; i++ )
    {
        double x = src_corners[i].x, y = src_corners[i].y;
	    double Z = 1./(H_combo->data.db[6]*x + H_combo->data.db[7]*y + H_combo->data.db[8]);
        double X = (H_combo->data.db[0]*x + H_combo->data.db[1]*y + H_combo->data.db[2])*Z;
        double Y = (H_combo->data.db[3]*x + H_combo->data.db[4]*y + H_combo->data.db[5])*Z;
        dst_corners[i] = cvPoint(cvRound(X), cvRound(Y));
    }
	
	//flush the list of corners and recompute the minimum bounding box
	cvClearSeq(cornerSequence);
	//cvSeqPushMulti(cornerSequence, img_corners, 4, CV_BACK);
	cvSeqPushMulti(cornerSequence, dst_corners, 4, CV_BACK);
	mbr = cvBoundingRect(cornerSequence, 0);
	
	cout<<"minimum bounding rectangle: {x: "<<mbr.x<<"  y: "<<mbr.y<<"  width: "<<mbr.width<<"  height: "<<mbr.height<<endl;



	// Draw the matching keypairs in each image and a line between them 
	vector<int> ptpairs;
    findPairs( objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors, ptpairs );
    for( i = 0; i < (int)ptpairs.size(); i += 2 )
    {
        CvSURFPoint* r1 = (CvSURFPoint*)cvGetSeqElem( objectKeypoints, ptpairs[i] );
        CvSURFPoint* r2 = (CvSURFPoint*)cvGetSeqElem( imageKeypoints, ptpairs[i+1] );
//        cvLine( correspond, cvPointFrom32f(r1->pt), cvPoint(cvRound(r2->pt.x), cvRound(r2->pt.y+object->height)), colors[8] );
    }

    //cvShowImage( "Object Correspond", correspond );
    for( i = 0; i < objectKeypoints->total; i++ )
    {
        CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( objectKeypoints, i );
        CvPoint center;
        int radius;
        center.x = cvRound(r->pt.x);
        center.y = cvRound(r->pt.y);
        radius = cvRound(r->size*1.2/9.*2);
        cvCircle( object_color, center, radius, colors[0], 1, 8, 0 );
    }


	//IplImage * rectifiedImage = cvCreateImage(cvGetSize(object), object->depth, object->nChannels);
	
	IplImage * rectifiedImage2 = cvCreateImage(cvSize(mbr.width, mbr.height), object->depth, object->nChannels);
	//IplImage * rectifiedImage3 = cvCreateImage(cvSize(mbr.width*2, mbr.height*2), color_object->depth, color_object->nChannels);
	//cvWarpPerspective(object, rectifiedImage, H, CV_INTER_LINEAR |  CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
	cvWarpPerspective(object, rectifiedImage2, H_combo, CV_INTER_LINEAR |  CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
	//cvWarpPerspective(color_object, rectifiedImage3, H_scale, CV_INTER_LINEAR |  CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

	//figure out how to place the images based on the recorded x/y offsets (if any) and mbr size

	
	cout<<"x offset: "<<x_offset<<"   y offset: "<<y_offset<<endl;

	//cvSetImageROI(rectifiedImage2, cvRect(0,0,mbr.width, mbr.height));

	int x1, y1, x2, y2, w, h;

	if(y_offset > 0) //image 2 is on top
	{ //works
		y2 = 0;
		y1 = abs(y_offset);
		cout<<"image 2 on top"<<endl;
		h = rectifiedImage2->height + y1;
	}
	else
	{
		y2 = abs(y_offset);
		y1 = 0;
		cout<<"image 1 on top"<<endl;
		h = max(image->height + y2, mbr.height);
	}

	if(x_offset >0) //image 2 is on left
	{
		x2 = 0;
		x1 = abs(x_offset);

		cout<<"image 2 on left"<<endl;
		w = rectifiedImage2->width + x1;//ceil(x1/2.0);
	}
	else
	{ //works
		x2 = abs(x_offset);
		x1 = 0;
		cout<<"image 1 on left"<<endl;
		w = max(image->width + x2, mbr.width);// ceil(x2/2.0);
	}

	CvSize outSize = cvSize(w, h);
	cout<<"outsize: "<<outSize.width<<" x "<<outSize.height<<endl;
	IplImage * comboImage = cvCreateImage(outSize, object->depth, object->nChannels);
	
	cvSetImageROI(comboImage, cvRect(x1, y1, rectifiedImage2->width, rectifiedImage2->height));
	cvCopy(rectifiedImage2, comboImage);
	cvSetImageROI(comboImage, cvRect(x2, y2, image->width, image->height));
	//cvCopy(image, comboImage);
	//cvAddWeighted(comboImage, 0.5, image, 0.5, 0.0, comboImage);
	cvMax(image, comboImage, comboImage);
	cvResetImageROI(comboImage);

    //cvShowImage( "Object", object_color );
	//cvSaveImage("object.png", object);
	//cvSaveImage("image.png", image);
	//cvSaveImage("blah.png", correspond);
	//cvSaveImage("warp.png", rectifiedImage);
	//cvSaveImage("warp2.png", rectifiedImage2);
	//cvSaveImage("warp3.png", rectifiedImage3);
	
	cvSaveImage(output_filename, comboImage );
    
	double t3 = (double)cvGetTickCount();
	printf( "Total elapsed time = %gms\n", (t3-t1) /(cvGetTickFrequency()*1000.));

    return 0;
}


