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
