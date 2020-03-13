// OCT 11, modification :
// 1.add degree as union type in MyMarker, 
// 2. MyMarker is alias as MyNode
// 3. just disable __SET_MARKER_DEGREE__ if you don't like these change

//last change; by PHC 20121127. update the swc and marker saving funcs

// modified from vaa3d/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2/my_surf_objs.h mzhu 05/23/2019

#ifndef MCP3D_VAA3D_MARKERS
#define MCP3D_VAA3D_MARKERS

#include <vector>
#include <list>
#include <map>
#include <iostream>
#include <fstream>
#include <cassert>
#include "common/mcp3d_macros.hpp"

using namespace std;

#define __SET_MARKER_DEGREE__
#define MARKER_BASE 1.0 // the basic offset of marker is 1.0, the marker coordinate will be converted when read and save


struct MyPoint
{
	int x;
	int y;
	int z;
	bool operator<(const MyPoint & other) const{
		if(z > other.z) return false;
		if(z < other.z) return true;
		if(y > other.y) return false;
		if(y < other.y) return true;
		if(x > other.x) return false;
		if(x < other.x) return true;
		return false;
	}
	MyPoint()
	{
		x = 0;
		y = 0;
		z = 0;
	}
	MyPoint(int _x, int _y, int _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}
};

struct MYXYZ {
   double x,y,z;
};

// root node with parent 0
struct MyMarker
{
	double x;
	double y;
	double z;
	union
	{
#ifdef __SET_MARKER_DEGREE__
		double degree;
#endif
		double radius;
	};
	int type;
	MyMarker* parent;
	MyMarker(){x=y=z=radius=0.0; type = 3; parent=0;}
	MyMarker(double _x, double _y, double _z) {x = _x; y = _y; z = _z; radius = 0.0; type = 3; parent = 0;}
	MyMarker(const MyMarker & v){x=v.x; y=v.y; z=v.z; radius = v.radius; type = v.type; parent = v.parent;}
	MyMarker(const MyPoint & v){x=v.x; y=v.y; z=v.z; radius = 0.0; type = 3; parent = 0;}
	MyMarker(const MYXYZ & v){x=v.x; y=v.y; z=v.z; radius = 0.0; type = 3; parent = 0;}

	double & operator [] (const int i) {
		assert(i>=0 && i <= 2);
		return (i==0) ? x : ((i==1) ? y : z);
	}

	bool operator<(const MyMarker & other) const{                                                                 
		if(z > other.z) return false;
		if(z < other.z) return true;
		if(y > other.y) return false;
		if(y < other.y) return true;
		if(x > other.x) return false;
		if(x < other.x) return true;    
		return false;
	}
	bool operator==(const MyMarker & other) const{                                                                 
		return (z==other.z && y==other.y && x==other.x);
	}
	bool operator!=(const MyMarker & other) const{                                                                 
		return (z!=other.z || y!=other.y || x!=other.x);
	}

	long long ind(long long sz0, long long sz01)
	{
		return ((long long)(z+0.5) * sz01 + (long long)(y+0.5)*sz0 + (long long)(x+0.5));
	}

    // offsets is in zyx order
    template <typename T>
    void operator-=(const vector<T> offsets)
    {
        MCP3D_ASSERT(offsets.size() == 3l)
        z -= (double)offsets[0];
        y -= (double)offsets[1];
        x -= (double)offsets[2];
    }
};

struct MyMarkerX : public MyMarker
{
	double feature;
	int seg_id;
	int seg_level;
	MyMarkerX() : MyMarker()
	{
		seg_id = -1;
		seg_level = -1;
		feature = 0.0;
	}
	MyMarkerX(MyMarker & _marker)
	{
		x = _marker.x; y = _marker.y; z = _marker.z; type = _marker.type; radius = _marker.radius;
		seg_id = -1; seg_level = -1; feature = 0.0;
	}
	MyMarkerX(double _x, double _y, double _z) : MyMarker(_x, _y, _z)
	{
		seg_id = -1;
		seg_level = -1;
		feature = 0.0;
	}
};
typedef MyMarker MyNode;

#define MidMarker(m1, m2) MyMarker(((m1).x + (m2).x)/2.0,((m1).y + (m2).y)/2.0,((m1).z + (m2).z)/2.0)


vector<MyMarker> readMarker_file(string marker_file);
bool readMarker_file(string marker_file, vector<MyMarker*> &markers);
bool saveMarker_file(string marker_file, vector<MyMarker> & out_markers);
bool saveMarker_file(string marker_file, vector<MyMarker> & outmarkers, list<string> & infostring);
bool saveMarker_file(string marker_file, vector<MyMarker*> & out_markers);
bool saveMarker_file(string marker_file, vector<MyMarker*> & out_markers, list<string> & infostring);

double dist(MyMarker a, MyMarker b);

vector<MyMarker*> getLeaf_markers(vector<MyMarker*> & inmarkers);
vector<MyMarker*> getLeaf_markers(vector<MyMarker*> & inmarkers, map<MyMarker *, int> & childs_num);

#endif  // MCP3D_VAA3D_MARKERS
