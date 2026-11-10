#include "myutils.h"
#include "npy.h"
#include <errno.h>
#include <string.h>

// Format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
static const char NPY_MAGIC[] = "\x93NUMPY";
static const uint NPY_MAGIC_LEN = 6;
static const uint NPY_ALIGN = 64;
static const uint NPY_GROWTH_AXIS_MAX_DIGITS = 21;

static bool IsLittleEndian()
	{
	const uint32 x = 1;
	return *(const byte *) &x == 1;
	}

// The header numpy writes: the dict, spare spaces for growing axis 0,
// then spaces and '\n' so that the data starts at a multiple of 64.
static void MakeHeader(uint Rows, uint Cols, string &Header)
	{
	char Dict[128];
	snprintf(Dict, sizeof(Dict),
	  "{'descr': '<f4', 'fortran_order': False, 'shape': (%u, %u), }",
	  Rows, Cols);
	Header = string(Dict);
	uint Digits = SIZE(to_string(Rows));
	if (Digits < NPY_GROWTH_AXIS_MAX_DIGITS)
		Header += string(NPY_GROWTH_AXIS_MAX_DIGITS - Digits, ' ');
	uint Used = NPY_MAGIC_LEN + 2 + 2 + SIZE(Header) + 1;
	uint PadLen = NPY_ALIGN - Used%NPY_ALIGN;
	Header += string(PadLen, ' ');
	Header += '\n';
	}

void WriteNpyFloat2D(const string &FileName, const float *Data,
  uint Rows, uint Cols)
	{
	if (!IsLittleEndian())
		Die("-postout needs a little-endian CPU");
	string Header;
	MakeHeader(Rows, Cols, Header);
	asserta(SIZE(Header) <= 65535);

	FILE *f = fopen(FileName.c_str(), "wb");
	if (f == 0)
		Die("Cannot create %s, errno=%d %s",
		  FileName.c_str(), errno, strerror(errno));

	byte Prefix[10];
	memcpy(Prefix, NPY_MAGIC, NPY_MAGIC_LEN);
	Prefix[6] = 1;	// major version
	Prefix[7] = 0;	// minor version
	uint HeaderLen = SIZE(Header);
	Prefix[8] = byte(HeaderLen & 0xff);
	Prefix[9] = byte(HeaderLen >> 8);

	size_t N = size_t(Rows)*size_t(Cols);
	bool Ok = fwrite(Prefix, 1, 10, f) == 10 &&
	  fwrite(Header.c_str(), 1, HeaderLen, f) == HeaderLen &&
	  fwrite(Data, sizeof(float), N, f) == N;
	if (fclose(f) != 0)
		Ok = false;
	if (!Ok)
		Die("Write failed %s, errno=%d %s",
		  FileName.c_str(), errno, strerror(errno));
	}
