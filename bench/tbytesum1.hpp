/* Test whether read(), fread(), or mmap() are faster.
   Elmar Klausmeier, 22-Nov-2015
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

// 16 GB, minimum size of one interval
//#define MAXBUF 16000000000
#define MAXBUF 16000000

uint64_t trivial(int a, int b) {
  return a + b;
}

uint64_t freadtst (char *pgname, char *fname) {
	char buf[MAXBUF];
	uint64_t i, n, result=0;	// integer definition for i&n crucial for performance
	FILE *fp = fopen(fname, "r");

	if (fp == NULL) {
		printf("%s: cannot open %s\n", pgname,fname);
		exit(2);
	}
	while ( (n = fread(buf,1,MAXBUF,fp)) > 0 )
		for (i=0; i<n; i++)
			result += buf[i];
	if (fclose(fp) != 0) {
		printf("%s: cannot close %s\n", pgname,fname);
		exit(3);
	}
	return result;
}


uint64_t ifstreamtst(char *pgname, char* fname, uint64_t size_bytes) {
  //inmem_ptr0_ = (VertexAttr*) malloc(mmap_length_);
  char buf[MAXBUF];
  uint64_t i, result=0;
  std::ifstream ifile(fname, std::ios::in | std::ios::binary); // ifstream is default read only mode
  // open input
  assert(ifile.is_open());
  assert(ifile.good());

  // read only needed Vertexes
  ifile.read((char*)buf, 
      size_bytes // number of bytes
  );

  // close file 
  ifile.close();

  // sum all bytes, just to prevent any compiler optimizations
  for (i=0; i<MAXBUF; i++)
    result += buf[i];
}

uint64_t preadtst (char *pgname, char *fname, uint64_t size_bytes) {
	int fd;
    //size_bytes *= 2<<5; // increase by factor 16
    char* buf = (char*) malloc(size_bytes);
	uint64_t i, result=0;
	struct stat sb;

	if ((fd = open(fname,O_RDONLY)) < 0) {
		printf("%s: cannot open %s\n", pgname,fname);
		exit(2);
	}

	if (fstat(fd, &sb) != 0) {
		printf("%s: cannot stat %s\n", pgname,fname);
		exit(4);
	}

    // using size_bytes to read not the file size sb.st_size
	if (pread(fd, buf, size_bytes, 0) == -1) {
		printf("%s: cannot pread %s\n", pgname,fname);
		exit(5);
	}

    // sum all bytes, just to prevent any compiler optimizations
	for (i=0; i<size_bytes; i++)
		result += buf[i];

	if (close(fd) < 0) {
		printf("%s: cannot close %s\n", pgname,fname);
		exit(3);
	}

    free(buf);
	return result;
}

uint64_t mmaptst (char *pgname, char *fname, uint64_t size_bytes) {
	int fd;
	char *p;
	uint64_t i, result=0;
	struct stat sb;
    //size_bytes *= 2<<5; // increase by factor 16

	if ((fd = open(fname,O_RDONLY)) < 0) {
		printf("%s: cannot open %s\n", pgname,fname);
		exit(2);
	}
	if (fstat(fd, &sb) != 0) {
		printf("%s: cannot stat %s\n", pgname,fname);
		exit(4);
	}

    // using MAX_BUF bytes to read not the file size sb.st_size
    // let mmap choose the location
	if ((p = (char*) mmap(NULL,size_bytes, PROT_READ,MAP_SHARED|MAP_POPULATE,fd,0)) == MAP_FAILED) {
		printf("%s: cannot memory-map %s\n", pgname,fname);
		exit(5);
	}

    // sum all bytes, just to prevent any compiler optimizations
	for (i=0; i<size_bytes; i++)
		result += p[i];

	if (close(fd) < 0) {
		printf("%s: cannot close %s\n", pgname,fname);
		exit(3);
	}
    munmap(p, size_bytes);
	return result;
}



uint64_t readtst (char *pgname, char *fname) {
	int fd;
	char buf[MAXBUF];
	uint64_t i, n, result=0;

	if ((fd = open(fname,O_RDONLY)) < 0) {
		printf("%s: cannot open %s\n", pgname,fname);
		exit(2);
	}

	while ( (n = read(fd,buf,MAXBUF)) > 0 )
		for (i=0; i<n; ++i)
			result += buf[i];

	if (close(fd) < 0) {
		printf("%s: cannot close %s\n", pgname,fname);
		exit(3);
	}
	return result;
}
