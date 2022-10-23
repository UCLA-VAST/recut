#include "markers.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

using namespace std;

vector<MyMarker> readMarker_file(filesystem::path marker_file, int marker_base) {
  vector<MyMarker> markers;
  markers.reserve(1000);
  ifstream ifs(marker_file);
  if (ifs.fail()) {
    cout << " unable to open marker file " << marker_file << endl;
    return markers;
  }
  set<MyMarker> marker_set;
  int count = 0;
  while (ifs.good()) {
    if (ifs.peek() == '#' || ifs.eof()) {
      ifs.ignore(1000, '\n');
      continue;
    }
    MyMarker marker;
    ifs >> marker.x;
    ifs.ignore(10, ',');
    ifs >> marker.y;
    ifs.ignore(10, ',');
    ifs >> marker.z;
    ifs.ignore(10, ',');
    ifs >> marker.radius;
    ifs.ignore(1000, '\n');

    marker.x -= marker_base;
    marker.y -= marker_base;
    marker.z -= marker_base;

    if (0 && marker_set.find(marker) != marker_set.end()) {
      cout << "omit duplicated marker" << markers.size()
           << " : x = " << marker.x << " y = " << marker.y
           << " z = " << marker.z << " r = " << marker.radius << endl;
    } else {
      markers.push_back(marker);
      marker_set.insert(marker);
      if (0)
        cout << "marker" << markers.size() << " : x = " << marker.x
             << " y = " << marker.y << " z = " << marker.z
             << " r = " << marker.radius << endl;
    }
    count++;
  }
  // cout<<count<<" markers loaded"<<endl;
  ifs.close();
  return markers;
}

bool saveMarker_file(string marker_file, vector<MyMarker> &outmarkers) {
  list<string> nullinfostr;
  return saveMarker_file(marker_file, outmarkers, nullinfostr);
}

bool saveMarker_file(string marker_file, vector<MyMarker> &outmarkers,
                     list<string> &infostring) {
  cout << "save " << outmarkers.size() << " markers to file " << marker_file
       << endl;
  ofstream ofs(marker_file.c_str());

  if (ofs.fail()) {
    cout << "open marker file error" << endl;
    return false;
  }

  list<string>::iterator it;
  for (it = infostring.begin(); it != infostring.end(); it++)
    ofs << *it << endl;

  ofs << "#x, y, z, radius" << endl;
  for (long unsigned int i = 0; i < outmarkers.size(); i++) {
    ofs << outmarkers[i].x + MARKER_BASE << "," << outmarkers[i].y + MARKER_BASE
        << "," << outmarkers[i].z + MARKER_BASE << "," << outmarkers[i].radius
        << endl;
  }
  ofs.close();
  return true;
}

bool saveMarker_file(string marker_file, vector<MyMarker *> &outmarkers) {
  list<string> nullinfostr;
  return saveMarker_file(marker_file, outmarkers, nullinfostr);
}

bool saveMarker_file(string marker_file, vector<MyMarker *> &outmarkers,
                     list<string> &infostring) {
  cout << "save " << outmarkers.size() << " markers to file " << marker_file
       << endl;
  ofstream ofs(marker_file.c_str());

  if (ofs.fail()) {
    cout << "open marker file error" << endl;
    return false;
  }

  list<string>::iterator it;
  for (it = infostring.begin(); it != infostring.end(); it++)
    ofs << *it << endl;

  ofs << "#x, y, z, radius" << endl;
  for (long unsigned int i = 0; i < outmarkers.size(); i++) {
    ofs << outmarkers[i]->x + MARKER_BASE << ","
        << outmarkers[i]->y + MARKER_BASE << ","
        << outmarkers[i]->z + MARKER_BASE << "," << outmarkers[i]->radius
        << endl;
  }
  ofs.close();
  return true;
}

double dist(MyMarker a, MyMarker b) {
  return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
              (a.z - b.z) * (a.z - b.z));
}

vector<MyMarker *> getLeaf_markers(vector<MyMarker *> &inmarkers) {
  set<MyMarker *> par_markers;
  vector<MyMarker *> leaf_markers;
  for (long unsigned int i = 0; i < inmarkers.size(); i++) {
    MyMarker *marker = inmarkers[i];
    if (marker->parent)
      par_markers.insert(marker->parent);
  }
  for (long unsigned int i = 0; i < inmarkers.size(); i++) {
    if (par_markers.find(inmarkers[i]) == par_markers.end())
      leaf_markers.push_back(inmarkers[i]);
  }
  par_markers.clear();
  return leaf_markers;
}

vector<MyMarker *> getLeaf_markers(vector<MyMarker *> &inmarkers,
                                   map<MyMarker *, int> &childs_num) {
  for (long unsigned int i = 0; i < inmarkers.size(); i++)
    childs_num[inmarkers[i]] = 0;

  vector<MyMarker *> leaf_markers;
  for (long unsigned int i = 0; i < inmarkers.size(); i++) {
    MyMarker *marker = inmarkers[i];
    MyMarker *parent = marker->parent;
    if (parent)
      childs_num[parent]++;
  }
  for (long unsigned int i = 0; i < inmarkers.size(); i++) {
    if (childs_num[inmarkers[i]] == 0)
      leaf_markers.push_back(inmarkers[i]);
  }
  return leaf_markers;
}
