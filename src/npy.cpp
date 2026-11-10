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

#ifdef _WIN32
#define npy_fseek	_fseeki64
#define npy_ftell	_ftelli64
#else
#define npy_fseek	fseeko
#define npy_ftell	ftello
#endif

static void SkipSpaces(const string &s, size_t &Pos)
	{
	while (Pos < s.size() && s[Pos] == ' ')
		++Pos;
	}

// Position just after "'Key':" and any spaces, or string::npos.
static size_t FindValue(const string &Header, const char *Key)
	{
	string Pattern = string("'") + Key + "':";
	size_t Pos = Header.find(Pattern);
	if (Pos == string::npos)
		return string::npos;
	Pos += Pattern.size();
	SkipSpaces(Header, Pos);
	return Pos;
	}

static bool ParseUint(const string &s, size_t &Pos, uint &Value)
	{
	uint64 v = 0;
	size_t Start = Pos;
	while (Pos < s.size() && isdigit((unsigned char) s[Pos]))
		{
		v = 10*v + uint64(s[Pos] - '0');
		if (v > UINT_MAX)
			return false;
		++Pos;
		}
	Value = uint(v);
	return Pos > Start;
	}

// Shape "(R, C)" or "(R, C,)"; false for any other number of axes.
static bool ParseShape2D(const string &Header, uint &R, uint &C)
	{
	size_t Pos = FindValue(Header, "shape");
	if (Pos == string::npos || Pos >= Header.size() || Header[Pos] != '(')
		return false;
	++Pos;
	SkipSpaces(Header, Pos);
	if (!ParseUint(Header, Pos, R))
		return false;
	SkipSpaces(Header, Pos);
	if (Pos >= Header.size() || Header[Pos] != ',')
		return false;
	++Pos;
	SkipSpaces(Header, Pos);
	if (!ParseUint(Header, Pos, C))
		return false;
	SkipSpaces(Header, Pos);
	if (Pos < Header.size() && Header[Pos] == ',')
		{
		++Pos;
		SkipSpaces(Header, Pos);
		}
	return Pos < Header.size() && Header[Pos] == ')';
	}

void ReadNpyFloat2D(const string &FileName, uint Rows, uint Cols,
  float *Data)
	{
	const char *FN = FileName.c_str();
	if (!IsLittleEndian())
		Die("Reading .npy needs a little-endian CPU");
	FILE *f = fopen(FN, "rb");
	if (f == 0)
		Die("Cannot open posterior file %s, errno=%d %s",
		  FN, errno, strerror(errno));

	byte Prefix[12];
	if (fread(Prefix, 1, 8, f) != 8 ||
	  memcmp(Prefix, NPY_MAGIC, NPY_MAGIC_LEN) != 0)
		Die("%s: not a .npy file", FN);
	uint Major = Prefix[6];
	uint64 HeaderLen = 0;
	uint64 Offset = 0;
	if (Major == 1)
		{
		if (fread(Prefix + 8, 1, 2, f) != 2)
			Die("%s: truncated .npy header", FN);
		HeaderLen = uint64(Prefix[8]) | (uint64(Prefix[9]) << 8);
		Offset = 10 + HeaderLen;
		}
	else if (Major == 2 || Major == 3)
		{
		if (fread(Prefix + 8, 1, 4, f) != 4)
			Die("%s: truncated .npy header", FN);
		HeaderLen = uint64(Prefix[8]) | (uint64(Prefix[9]) << 8) |
		  (uint64(Prefix[10]) << 16) | (uint64(Prefix[11]) << 24);
		Offset = 12 + HeaderLen;
		}
	else
		Die("%s: .npy format version %u not supported", FN, Major);
	if (HeaderLen > 1000000)
		Die("%s: .npy header too long", FN);

	string Header(size_t(HeaderLen), ' ');
	if (fread(&Header[0], 1, size_t(HeaderLen), f) != size_t(HeaderLen))
		Die("%s: truncated .npy header", FN);

	size_t Pos = FindValue(Header, "descr");
	if (Pos == string::npos ||
	  (Header.compare(Pos, 5, "'<f4'") != 0 &&
	   Header.compare(Pos, 5, "\"<f4\"") != 0))
		Die("%s: dtype must be little-endian float32 ('<f4'), header %s",
		  FN, Header.c_str());
	Pos = FindValue(Header, "fortran_order");
	if (Pos == string::npos || Header.compare(Pos, 5, "False") != 0)
		Die("%s: must be C order (fortran_order False)", FN);
	uint R = 0, C = 0;
	if (!ParseShape2D(Header, R, C))
		Die("%s: shape must have 2 axes, header %s", FN, Header.c_str());
	if (R != Rows || C != Cols)
		Die("%s: shape (%u, %u), expected (%u, %u)", FN, R, C, Rows, Cols);

	uint64 Bytes = uint64(Rows)*uint64(Cols)*sizeof(float);
	if (npy_fseek(f, 0, SEEK_END) != 0)
		Die("%s: seek failed", FN);
	uint64 FileSize = uint64(npy_ftell(f));
	if (FileSize != Offset + Bytes)
		Die("%s: file size %llu, expected %llu for shape (%u, %u)",
		  FN, (unsigned long long) FileSize,
		  (unsigned long long) (Offset + Bytes), Rows, Cols);
	if (npy_fseek(f, (long long) Offset, SEEK_SET) != 0)
		Die("%s: seek failed", FN);
	size_t N = size_t(Rows)*size_t(Cols);
	if (fread(Data, sizeof(float), N, f) != N)
		Die("%s: read failed", FN);
	fclose(f);
	}
