#include "muscle.h"
#include "mpcflat.h"
#include "npy.h"
#include <errno.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
static int MkDir(const string &Dir) { return _mkdir(Dir.c_str()); }
#else
#include <sys/stat.h>
static int MkDir(const string &Dir) { return mkdir(Dir.c_str(), 0777); }
#endif

// Posteriors from / to files. Format (PLAN_posterior_input.md, section 2):
//   pair_map.tsv  family row_label col_label LX LY file transpose
//   file          float32 .npy, relative to the posterior directory,
//                 shape (LX, LY), or (LY, LX) if transpose is 1.
// row_label is the sequence that comes first in the input (MUSCLE's X).

static const char *POST_MAP_HEADER =
  "family\trow_label\tcol_label\tLX\tLY\tfile\ttranspose";

static void MakeDir(const string &Dir)
	{
	if (MkDir(Dir) != 0 && errno != EEXIST)
		Die("Cannot create directory %s, errno=%d %s",
		  Dir.c_str(), errno, strerror(errno));
	}

static void CheckMapLabel(const string &Label, const char *What)
	{
	if (Label.empty())
		Die("%s: empty label", What);
	if (Label.find('\t') != string::npos)
		Die("%s: label contains a tab >%s", What, Label.c_str());
	}

// Before the OpenMP loop: create DIR and DIR/pairs, write DIR/seqs.fasta
// (the parsed input, all sequences in input order, one line each).
void MPCFlat::PostOutBegin()
	{
	asserta(m_OriginalInputSeqs != 0);
	CheckMapLabel(m_PostFamily, "-postfamily");
	MakeDir(m_PostOutDir);
	MakeDir(m_PostOutDir + "/pairs");

	const string FileName = m_PostOutDir + "/seqs.fasta";
	FILE *f = CreateStdioFile(FileName);
	const uint SeqCount = m_OriginalInputSeqs->GetSeqCount();
	for (uint i = 0; i < SeqCount; ++i)
		{
		const Sequence *Seq = m_OriginalInputSeqs->GetSequence(i);
		const string &Label = Seq->GetLabel();
		CheckMapLabel(Label, "-postout");
		fprintf(f, ">%s\n", Label.c_str());
		WriteStdioFile(f, Seq->GetCharPtr(), Seq->GetLength());
		fputc('\n', f);
		}
	CloseStdioFile(f);

	m_PostOutMapLines.clear();
	m_PostOutMapLines.resize(SIZE(m_Pairs));
	}

// In the OpenMP loop: one file and one map line per pair.
void MPCFlat::SavePost(uint PairIndex, const string &LabelX,
  const string &LabelY, uint GSIX, uint GSIY, const float *Post,
  uint LX, uint LY)
	{
	asserta(GSIX < GSIY);
	asserta(PairIndex < SIZE(m_PostOutMapLines));
	const string File = "pairs/" + to_string(GSIX) + "_" +
	  to_string(GSIY) + ".npy";
	WriteNpyFloat2D(m_PostOutDir + "/" + File, Post, LX, LY);
	m_PostOutMapLines[PairIndex] = m_PostFamily + "\t" + LabelX + "\t" +
	  LabelY + "\t" + to_string(LX) + "\t" + to_string(LY) + "\t" +
	  File + "\t0";
	}

// After the OpenMP loop: the map, in PairIndex order.
void MPCFlat::PostOutEnd() const
	{
	const string FileName = m_PostOutDir + "/pair_map.tsv";
	FILE *f = CreateStdioFile(FileName);
	fprintf(f, "%s\n", POST_MAP_HEADER);
	for (uint i = 0; i < SIZE(m_PostOutMapLines); ++i)
		{
		const string &Line = m_PostOutMapLines[i];
		asserta(!Line.empty());
		fprintf(f, "%s\n", Line.c_str());
		}
	CloseStdioFile(f);
	}

static bool ParseMapUint(const string &s, uint &Value)
	{
	if (s.empty() || s.size() > 9)
		return false;
	uint v = 0;
	for (size_t i = 0; i < s.size(); ++i)
		{
		if (!isdigit((unsigned char) s[i]))
			return false;
		v = 10*v + uint(s[i] - '0');
		}
	Value = v;
	return true;
	}

// Fields split on tabs, keeping empty fields.
static void SplitTabs(const string &Line, vector<string> &Fields)
	{
	Fields.clear();
	size_t Start = 0;
	for (;;)
		{
		size_t Tab = Line.find('\t', Start);
		if (Tab == string::npos)
			{
			Fields.push_back(Line.substr(Start));
			return;
			}
		Fields.push_back(Line.substr(Start, Tab - Start));
		Start = Tab + 1;
		}
	}

// Before the OpenMP loop: read the lines of family m_PostFamily from the
// map, and check (serially, so errors are reported cleanly) that every
// pair MUSCLE will need has a line, with row = the sequence that comes
// first in the input and the lengths MUSCLE parsed.
void MPCFlat::PostInBegin()
	{
	const string &FN = m_PostMapFileName;
	const string &Fam = m_PostFamily;
	m_PostMap.clear();
	FILE *f = fopen(FN.c_str(), "rb");
	if (f == 0)
		Die("-postmap: cannot open %s, errno=%d %s",
		  FN.c_str(), errno, strerror(errno));

	string Line;
	vector<string> Fields;
	uint LineNr = 0;
	uint FamLineCount = 0;
	bool Eof = false;
	while (!Eof)
		{
		Line.clear();
		int c;
		while ((c = fgetc(f)) != EOF && c != '\n')
			Line.push_back(char(c));
		if (c == EOF)
			{
			Eof = true;
			if (Line.empty())
				break;
			}
		++LineNr;
		if (!Line.empty() && Line[Line.size()-1] == '\r')
			Line.resize(Line.size() - 1);
		if (LineNr == 1)
			{
			if (Line != POST_MAP_HEADER)
				Die("-postmap %s: header must be '%s'",
				  FN.c_str(), POST_MAP_HEADER);
			continue;
			}
		SplitTabs(Line, Fields);
		if (SIZE(Fields) != 7)
			Die("-postmap %s line %u: %u fields, expected 7",
			  FN.c_str(), LineNr, SIZE(Fields));
		if (Fields[0] != Fam)
			continue;
		++FamLineCount;
		PostMapEntry E;
		if (!ParseMapUint(Fields[3], E.LX) || !ParseMapUint(Fields[4], E.LY))
			Die("-postmap %s line %u: bad LX or LY", FN.c_str(), LineNr);
		if (Fields[6] == "1")
			E.Transpose = true;
		else if (Fields[6] != "0")
			Die("-postmap %s line %u: transpose must be 0 or 1",
			  FN.c_str(), LineNr);
		if (Fields[5].empty())
			Die("-postmap %s line %u: empty file", FN.c_str(), LineNr);
		E.File = Fields[5];
		const string &Row = Fields[1];
		const string &Col = Fields[2];
		pair<string, string> Key(Row, Col);
		pair<string, string> Swapped(Col, Row);
		if (m_PostMap.find(Key) != m_PostMap.end() ||
		  m_PostMap.find(Swapped) != m_PostMap.end())
			Die("-postmap %s line %u: family %s has pair %s, %s twice",
			  FN.c_str(), LineNr, Fam.c_str(), Row.c_str(), Col.c_str());
		m_PostMap[Key] = E;
		}
	fclose(f);

	const uint PairCount = SIZE(m_Pairs);
	if (FamLineCount == 0 && PairCount > 0)
		Die("-postmap %s has no lines for family %s",
		  FN.c_str(), Fam.c_str());

	for (uint PairIndex = 0; PairIndex < PairCount; ++PairIndex)
		{
		const pair<uint, uint> &Pair = GetPair(PairIndex);
		const string LabelX = string(GetLabel(Pair.first));
		const string LabelY = string(GetLabel(Pair.second));
		map<pair<string, string>, PostMapEntry>::const_iterator p =
		  m_PostMap.find(pair<string, string>(LabelX, LabelY));
		if (p == m_PostMap.end())
			{
			if (m_PostMap.find(pair<string, string>(LabelY, LabelX)) !=
			  m_PostMap.end())
				Die("-postmap %s, family %s: pair has row_label %s and "
				  "col_label %s, but row_label must be the sequence that "
				  "comes first in the input (%s)", FN.c_str(), Fam.c_str(),
				  LabelY.c_str(), LabelX.c_str(), LabelX.c_str());
			Die("-postmap %s, family %s: no line for pair %s, %s",
			  FN.c_str(), Fam.c_str(), LabelX.c_str(), LabelY.c_str());
			}
		const PostMapEntry &E = p->second;
		uint LX = GetSeqLength(Pair.first);
		uint LY = GetSeqLength(Pair.second);
		if (E.LX != LX || E.LY != LY)
			Die("-postmap %s, family %s, pair %s, %s: lengths %u, %u in the "
			  "map, %u, %u in the input", FN.c_str(), Fam.c_str(),
			  LabelX.c_str(), LabelY.c_str(), E.LX, E.LY, LX, LY);
		}

	ProgressLog("Posteriors: %u pairs from %s, %u unused lines of family %s\n",
	  PairCount, FN.c_str(), SIZE(m_PostMap) - PairCount, Fam.c_str());
	}

// NaN / inf test on the bits: the build uses -ffast-math, under which the
// compiler may assume std::isfinite is always true.
static bool IsFiniteBits(float x)
	{
	uint32 Bits;
	memcpy(&Bits, &x, sizeof(Bits));
	return (Bits & 0x7f800000u) != 0x7f800000u;
	}

// In the OpenMP loop: the dense posterior of one pair, LX*LY, X = rows.
// Values must be finite and non-negative; values below MIN_SPARSE_PROB
// are set to 0, as CalcPostFlat does.
float *MPCFlat::LoadPost(const string &LabelX, const string &LabelY,
  uint LX, uint LY) const
	{
	map<pair<string, string>, PostMapEntry>::const_iterator p =
	  m_PostMap.find(pair<string, string>(LabelX, LabelY));
	asserta(p != m_PostMap.end());
	const PostMapEntry &E = p->second;
	asserta(E.LX == LX && E.LY == LY);
	const string FileName = (E.File[0] == '/' ? E.File :
	  m_PostInDir + "/" + E.File);

	const uint FileRows = (E.Transpose ? LY : LX);
	const uint FileCols = (E.Transpose ? LX : LY);
	float *Post = AllocPost(LX, LY);
	float *Buffer = (E.Transpose ? myalloc(float, size_t(LX)*LY) : Post);
	ReadNpyFloat2D(FileName, FileRows, FileCols, Buffer);

	const size_t N = size_t(LX)*size_t(LY);
	for (size_t k = 0; k < N; ++k)
		{
		float P = Buffer[k];
		if (!IsFiniteBits(P))
			Die("%s: non-finite value at [%u, %u]", FileName.c_str(),
			  uint(k/FileCols), uint(k%FileCols));
		if (P < 0)
			Die("%s: negative value %g at [%u, %u]", FileName.c_str(), P,
			  uint(k/FileCols), uint(k%FileCols));
		if (P < MIN_SPARSE_PROB)
			Buffer[k] = 0;
		}

	if (E.Transpose)
		{
		for (uint y = 0; y < LY; ++y)
			for (uint x = 0; x < LX; ++x)
				Post[size_t(x)*LY + y] = Buffer[size_t(y)*LX + x];
		myfree(Buffer);
		}
	return Post;
	}
