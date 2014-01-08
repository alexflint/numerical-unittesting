Unit-testing numeric software
===

Unit testing aims to make software development more efficient and
robust. This document contains some points relevant to unit testing
numerical software.

First, read this general-purpose introduction to unittesting with
gtest:

- https://code.google.com/p/googletest/wiki/Primer

- https://code.google.com/p/googletest/wiki/AdvancedGuide


#### Prefer `EXPECT_*` to `ASSERT_*`

If a test such as `EXPECT_EQ(x,2)` fails then gtest will generate an
error message and the continue executing the current test. This should
be used by default because such unittests will generate more
informative error messages. Use `ASSERT_*` to terminate the unittest
if the condition fails. This may be desired if further code will
crash unless the condition is satisfied. Here are examples of both situations:

```
TEST(sqrt) {
	EXPECT_FLOAT_EQ(sqrt(4.0), 2.0);  // if this fails then we can still run the 
 // next test safely
	EXPECT_FLOAT_EQ(sqrt(100.0), 10.0);
}
```

```
TEST(GetRowPointer) {
	unsigned char image[] = { … }
	unsigned char* row = GetRowPointer(image, 130)
	ASSERT_NOT_NULL(row);  	// if this fails then this unittest must 
	// halt immediately
	EXPECT_EQ(row[0], 127);
	EXPECT_EQ(row[1], 131);
}
```

#### Use built-in macros instead of raw EXPECT()

Gtest provides a number of built-in macros for testing common
conditions, including `EXPECT_EQ`, `EXPECT_LT`, `EXPECT_STREQ`, and so
on. Using `EXPECT_EQ(a,b)` in place of `EXPECT(a == b)` allows gtest
to generate an output message that includes the current value of `a`
and `b`. This makes it much faster to track down errors.

Good:

```
EXPECT_FLOAT_EQ(expected_value, actual_value, tolerance)
```

Will produce error output that says something like “expected 21.5,
actual -21.5, tolerance was 1e-9”. This may be enough information to
fix the problem immediately.

Bad:

```
EXPECT(abs(expected_value - actual_value) < tolerance)
```

Will produce error output that essentially just says “test
failed”. This will force you to go into the code, fire up a debugger
or add printfs, recompile, etc etc.

#### If you must use a custom predicate, use `EXPECT_PRED1` or `EXPECT_PRED2`

Gtest does not have built-in macros for every possible comparison, so
sometimes you will be forced to use a custom expectation. For example,
there is no built-in gtest macro for parity checks. For this purpose
you can use `EXPECT_PRED1` and `EXPECT_PRED2`.

Good:

```
bool IsEven(int a) {
	return a % 2 == 0;
}
TEST(FunctionThatShouldReturnEvenValues) {
	int x = FunctionThatShouldReturnEvenValues();
	EXPECT_PRED1(&IsEven, x)
}
```

Will produce output like “IsEven(x) failed for x=123456”

Bad:

```
TEST(FunctionThatShouldReturnEvenValues) {
	int x = FunctionThatShouldReturnEvenValues();
	EXPECT(IsEven(x))
}
```

Will produce output like “IsEven(x) failed”

#### Test small components Don’t try to test large components all at once.

Each unittest should test only one function. Where possible, factor
code out of monolithic functions/classes and into small, well-defined
free functions in order to make it testable. Test individual functions
that have well-defined inputs and outputs.

```
TEST(RadialDistortionWithJacobian_Ordinary) {
    double k[] = { -0.2, 0.06, 0.0 };
    double p[] = { -1.5, -2.5 };
    double xd[] = { 0.3, 0.2 };
    double xc[2], J[4];
    RadialDistortionWithJacobian(xc, J, xd, k, p);
    double J_expected[] = { 0.88401552, -0.07730106, -0.07730106,  0.85240121 };
    CHECK_CLOSE_FRACTION(xc[0], 0.28926505, 1e-6);
    EXPECT_NEAR(xc[1], 0.18300459, 1e-6);
    CHECK_CLOSE_ARRAYS(J, J_expected, 4, 1e-8);
}
```

#### Do not use random number generators inside unittests

Unittests should be deterministic and therefore repeatabable. If you
use a random number generator inside unittests then the tests may
sporadically pass or fail, and when they do fail they may not produce
consistent failures.

#### Even if you set the random seed to some deterministic value, do not use random number generators inside unittests

When a test fails, it should generate error messages that make it as
easy as possible to track down the problem. It is *not* sufficient for
a test to merely detect errors, even if it does so reliably and
deterministically. In addition unittests should *elucidate* the cause
of the problem as far as possible. If your test uses random data (even
if it uses the *same* randomly generated data each time) then the
error messages generated on failure will be useless for tracking down
problems.


#### Use test fixtures to set up complicated data structures

See https://code.google.com/p/googletest/wiki/Primer#Test_Fixtures:_Using_the_Same_Data_Configuration_for_Multiple_Te

#### Example: testing a specialized tridiagonal cholesky solver

```
class MatrixFixture : public  ::testing::Test {
    static double As[];
    static double Bs[];
    static double R[];
    static double expected_soln[];
};

double MatrixFixture::As[] = {
    22, 9, 9,
    9, 24, 15,
    9, 15, 18,
    16, 9, 13,
    9, 24, 9,
    13, 9, 22,
    10, 5, 2,
    5, 18, 8,
    2, 8, 22,
    22, 10, 13,
    10, 14, 11,
    13, 11, 18,
    24, 8, 7,
    8, 20, 11, 
    7, 11, 16
};

double MatrixFixture::Bs[] = {
    0, 6, 4,
    1, 3, 6,
    5, 7, 9,
    7, 4, 5,
    0, 7, 8,
    7, 0, 2,
    5, 5, 8,
    9, 3, 7,
    6, 9, 1,
    8, 2, 4,
    6, 1, 6,
    5, 4, 7
};

double MatrixFixture::R[] = {
    291, 192, 173, 344,
    276, 268, 140, 384,
    318, 318, 236, 394,
    277, 410, 366, 337,
    401, 474, 463, 446,
    340, 411, 379, 368,
    178, 310, 216, 223,
    267, 461, 290, 310,
    349, 425, 303, 315,
    301, 484, 279, 331,
    269, 350, 244, 295,
    260, 410, 279, 348,
    188, 253, 289, 331,
    182, 156, 232, 321,
    208, 205, 234, 307
};

double MatrixFixture::expected_soln[] = {
    7, 2, 3, 8,
    3, 5, 0, 8,
    6, 5, 3, 6,
    3, 10, 8, 9,
    6, 7, 10, 7,
    5, 4, 5, 0,
    3, 5, 4, 6,
    4, 10, 5, 5,
    8, 6, 4, 6,
    4, 8, 1, 1,
    4, 4, 3, 2,
    2, 5, 4, 6,
    2, 4, 7, 7,
    4, 2, 5, 8,
    6, 4, 5, 7
};

TEST_F(TestSolve, MatrixFixture) {
    SymmetricTridiagonalCholesky<double,3> chol;
    chol.compute(As, Bs, 5);
    
    double soln[3*5*4];
    chol.solve(soln, R, 4);
    
    EXPECT_NEAR_ARRAYS(soln, expected_soln, 3*5*4, 1e-8);
}
```

#### Example: testing feature tracker

Here is one test (of many) that might be used to test a feature
tracker. We set up two black images with one white spot in each image
and check that the feature tracker identifies and tracks the white
dot.

```
TEST(FeatureTracker) {
  unsigned char* image0 = new unsigned char[kWidth*kHeight];
  unsigned char* image1 = new unsigned char[kWidth*kHeight];

	// Initialize both images to black
  memset(image0, 0, kWidth*kHeight);
  memset(image1, 0, kWidth*kHeight);

	// Create one white dot in each image
  image0[ 50*kWidth + 40 ] = 255;
  image1[ 52*kWidth + 37 ] = 255;

  Eigen::Matrix2Xd features0;
  Eigen::Matrix2Xd features1;
  std::vector<int> feature_ids;

  CvFlyBy::FeatureTracker tracker;
  tracker.Init(kWidth, kHeight, 200, .1, 0.);

  tracker.AddImage(image0, &features1, &features0, &feature_ids);
  tracker.AddImage(image1, &features1, &features0, &feature_ids);

	EXPECT_EQ(feature_ids.size(), 1);
	EXPECT_EQ(features0.cols(), 1);
	EXPECT_EQ(features1.cols(), 1);
	EXPECT_FLOAT_EQ(features0(0,0), 40.);
	EXPECT_FLOAT_EQ(features0(0,1), 50.);
	EXPECT_FLOAT_EQ(features0(0,0), 37.);
	EXPECT_FLOAT_EQ(features0(0,1), 52.);
}
```

#### Example: testing a feature track index

In our SLAM system, `SlidingWindow` is a data structure responsible
for stitching a series of frame-to-frame correspondences from the
feature tracker into a list of feature tracks. Rather than putting raw
images into this data structure and testing the results of the entire
system (which would be more like an integration test than a unittest),
we set up the `SlidingWindow` to accept feature correspondences
directly.

```
TEST(AddOneFrame) {    
    const observation_t K[] = {
        .5, 0., 0.,
        0., .25, 0.,
        0., 0., 1.
    };
    
    PerspectiveLensModel<observation_t> lensModel;
    lensModel.setK(K);
    
    std::vector<Point2D<observation_t> > locations;
    std::vector<int> track_indices;
    std::vector<FeatureMatch> matches;
    
    locations.resize(2);
    locations[0].assign(1.5, 2.5);
    locations[1].assign(3.5, 4.5);
    
    SlidingWindow<Frame> index;
    index.Init(&lensModel, 10, 10, false);
    
    index.AddFrame(123, locations, matches, track_indices);
    
    ASSERT_EQ(track_indices.size(), 2);
    EXPECT_EQ(track_indices[0], 0);
    EXPECT_EQ(track_indices[1], 1);
    
    EXPECT_EQ(index.GetTrack(0).mTrackID, 0);
    EXPECT_EQ(index.GetTrack(0).mFirstFrameID, 123);
    EXPECT_EQ(index.GetTrack(0).mLastFrameID, 123);
    EXPECT_EQ(index.GetTrack(0).Length(), 1);
    EXPECT_EQ(index.GetTrack(0).mObservations[0].mFrameID, 123);
    EXPECT_NEAR(index.GetTrack(0).mObservations[0].mImageLocation[0], 1.5, 1e-6);
    EXPECT_NEAR(index.GetTrack(0).mObservations[0].mImageLocation[1], 2.5, 1e-6);
    EXPECT_NEAR(index.GetTrack(0).mObservations[0].mCalibratedLocation[0], 3., 1e-6);
    EXPECT_NEAR(index.GetTrack(0).mObservations[0].mCalibratedLocation[1], 10., 1e-6);
    
    EXPECT_EQ(index.GetTrack(1).mTrackID, 1);
    EXPECT_EQ(index.GetTrack(1).mFirstFrameID, 123);
    EXPECT_EQ(index.GetTrack(1).mLastFrameID, 123);
    EXPECT_EQ(index.GetTrack(1).Length(), 1);
    EXPECT_EQ(index.GetTrack(1).mObservations[0].mFrameID, 123);
    EXPECT_NEAR(index.GetTrack(1).mObservations[0].mImageLocation[0], 3.5, 1e-6);
    EXPECT_NEAR(index.GetTrack(1).mObservations[0].mImageLocation[1], 4.5, 1e-6);
    EXPECT_NEAR(index.GetTrack(1).mObservations[0].mCalibratedLocation[0], 7., 1e-6);
    EXPECT_NEAR(index.GetTrack(1).mObservations[0].mCalibratedLocation[1], 18., 1e-6);
}

TEST(AddTwoFrames) {
    const observation_t K[] = {
        .5, 0., 0.,
        0., .25, 0.,
        0., 0., 1.
    };
    
    PerspectiveLensModel<observation_t> lensModel;
    lensModel.setK(K);
    
    std::vector<Point2D<observation_t> > locations;
    std::vector<FeatureMatch> matches;
    std::vector<int> first_track_indices;
    std::vector<int> second_track_indices;
    
    Frame frame;
    locations.resize(2);
    locations[0].assign(1.5, 2.5);
    locations[1].assign(3.5, 4.5);
    
    SlidingWindow<Frame> index;
    index.Init(&lensModel, 10, 10, false);
    
    std::vector<precision_t> v1;
    std::vector<precision_t> v2;
    index.Get2DTo2DCorrespondencesForEpipolar(v1, v2, 1);
    std::vector<int> v3;
    index.ComputeCovisibilityVector(v3);
    
    // In the first frame, there are two new features
    index.AddFrame(100, locations, matches, first_track_indices);
    
    // In the second frame, one feature is a match and one is new
    locations[0].assign(25., 50.);
    locations[1].assign(125., 150.);
    matches.push_back(FeatureMatch(0, 1));
    index.AddFrame(200, locations, matches, second_track_indices);
    
    ASSERT_EQ(second_track_indices.size(), 2);
    EXPECT_EQ(second_track_indices[0], 1);  // the index for the matched feature
    EXPECT_EQ(second_track_indices[1], 2);  // the index for the new feature
    
    EXPECT_EQ(index.GetTrack(1).mFirstFrameID, 100);
    EXPECT_EQ(index.GetTrack(1).mLastFrameID, 200);
    EXPECT_EQ(index.GetTrack(1).Length(), 2);
    EXPECT_EQ(index.GetTrack(1).mObservations[0].mFrameID, 100);
    EXPECT_NEAR(index.GetTrack(1).mObservations[0].mImageLocation[0], 3.5, 1e-6);
    EXPECT_NEAR(index.GetTrack(1).mObservations[0].mImageLocation[1], 4.5, 1e-6);
    EXPECT_EQ(index.GetTrack(1).mObservations[1].mFrameID, 200);
    EXPECT_NEAR(index.GetTrack(1).mObservations[1].mImageLocation[0], 25., 1e-6);
    EXPECT_NEAR(index.GetTrack(1).mObservations[1].mImageLocation[1], 50., 1e-6);
    
    EXPECT_EQ(index.GetTrack(2).mFirstFrameID, 200);
    EXPECT_EQ(index.GetTrack(2).mLastFrameID, 200);
    EXPECT_EQ(index.GetTrack(2).Length(), 1);
    EXPECT_EQ(index.GetTrack(2).mObservations[0].mFrameID, 200);
    EXPECT_NEAR(index.GetTrack(2).mObservations[0].mImageLocation[0], 125., 1e-6);
    EXPECT_NEAR(index.GetTrack(2).mObservations[0].mImageLocation[1], 150., 1e-6);
}
```

#### Example: testing a bundle adjuster

To create bundle adjustment unittests, we first create a test fixture
that sets up a simple network with 3 cameras and 10 structure points.

```
// This fixture sets up a VisualInertialBundle with 3 cameras and 10 point
class ThreeCameraFixture : public ::testing::Test {
protected:
    static int nc;
    static int nt;
    static state_t Rs[];
    static state_t ts[];
    static state_t structure[];
    static state_t measurements[];
    static mask_t measurement_mask[];
    
    VisualInertialBundle bundle;
    VisionMeasurements msm;
    
    SetUp() : bundle(nt, nc) {
        CopyVector(bundle.orientation_data(), Rs, nc*9);
        CopyVector(bundle.position_data(), ts, nc*3);
        CopyVector(bundle.structure_data(), structure, nt*3);
        for (int i = 0; i < 10; i++) {
            CopyVector(msm.measurements_for_track(i), measurements+i*6, nc*2);
            CopyVector<unsigned char>(msm.mask_for_track(i), measurement_mask+i*3, nc);
        }
    }
};

int ThreeCameraFixture::nc = 3;
int ThreeCameraFixture::nt = 10;

state_t ThreeCameraFixture::Rs[] = {
    0.22629564095, -0.183007919658, 0.956712278707,
    0.956712278707, 0.22629564095, -0.183007919658,
    -0.183007919658, 0.956712278707, 0.22629564095,
    
    -0.474921441458, 0.329441812987, 0.816037815483,
    0.850495340179, 0.410031423417, 0.329441812987,
    -0.2260692389, 0.850495340179, -0.474921441458,
    
    -0.694920557641, 0.713520990528, 0.0892928588619,
    -0.192006972792, -0.303785044339, 0.933192353824,
    0.692978167742, 0.631349699384, 0.34810747783
};

state_t ThreeCameraFixture::ts[] = {
    1.0, 2.0, 3.0,
    -1.0, 2.0, -1.0,
    -1.0, 3.0, 0.0,
};

//
state_t ThreeCameraFixture::structure[] = {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9, 10, 11,
    12, 13, 14,
    15, 16, 17,
    18, 19, 20,
    21, 22, 23,
    24, 25, 26,
    27, 28, 29,
};

state_t ThreeCameraFixture::measurements[] = {
    0.0, 0.0,
    20.0, 21.0,
    40.0, 41.0,
    
    2.0, 3.0,
    22.0, 23.0,
    42.0, 43.0,
    
    0.0, 0.0,  // missing
    0.0, 0.0,  // missing
    0.0, 0.0,  // missing
    
    6.0, 7.0,
    26.0, 27.0,
    46.0, 47.0,
    
    8.0, 9.0,
    28.0, 29.0,
    48.0, 49.0,
    
    10.0, 11.0,
    0.0, 0.0,   // missing
    50.0, 51.0,
    
    12.0, 13.0,
    32.0, 33.0,
    52.0, 53.0,
    
    0.0, 0.0,   // missing
    34.0, 35.0,
    54.0, 55.0,
    
    16.0, 17.0,
    36.0, 37.0,
    0.0, 0.0,   // missing
    
    18.0, 19.0,
    38.0, 39.0,
    58.0, 59.0,
};

mask_t ThreeCameraFixture::measurement_mask[] = {
    false, true, true,
    true, true, true,
    false, false, false,
    true, true, true,
    true, true, true,
    true, false, true,
    true, true, true,
    false, true, true,
    true, true, false,
    true, true, true,
};
```

Now we can write a series of tests against this fixture:

```
TEST_F(VisionNormalEquations, ThreeCameraFixture) {
    ba.configure(bundle, &vision_msm, &inertial_msm);
    
    ba.set_initial_damping(0.);
    ba.set_damping_strategy(DampingStrategy::LEVENBERG_MARQUARDT);
    ba.prepare_to_optimize();
    
    ba.compute_vision_terms();
    ba.initialize_normal_equations();
    ba.add_vision_normal_equations();
    
#   include "data/VisionNormalEquations.inc"
    
    WriteMatrixToFile(ba.m_normal_eqns_lhs.data(), 3*15, 3*15, "/tmp/normal_eqns_lhs.txt");
    WriteMatrixToFile(ba.m_normal_eqns_rhs.data(), 3*15, 1, "/tmp/normal_eqns_rhs.txt");
    
    EXPECT_NEAR_ARRAYS(ba.m_normal_eqns_lhs.data(), H_expected, 3*3*15*15, 1e-4);
    EXPECT_NEAR_ARRAYS(ba.m_normal_eqns_rhs.data(), g_expected, 3*15,      1e-4);
    EXPECT_NEAR(ba.compute_vision_cost(bundle), cost_expected, 1e-4);
}

TEST_F(InertialNormalEquations, ThreeCameraFixture) {
    ba.configure(bundle, &vision_msm, &inertial_msm);
    
    ba.set_initial_damping(0.);
    ba.set_damping_strategy(DampingStrategy::LEVENBERG_MARQUARDT);
    ba.prepare_to_optimize();
    
    ba.compute_inertial_terms();
    ba.initialize_normal_equations();
    ba.add_inertial_normal_equations();
    
#   include "data/InertialNormalEquations.inc"
    EXPECT_NEAR_ARRAYS(ba.m_normal_eqns_lhs.data(), H_expected, 3*3*15*15, 1e-5);
    EXPECT_NEAR_ARRAYS(ba.m_normal_eqns_rhs.data(), g_expected, 3*15, 1e-5);
    EXPECT_NEAR(ba.compute_inertial_cost(bundle), cost_expected, 1e-4);
}

TEST_F(JointNormalEquations, ThreeCameraFixture) {
    ba.configure(bundle, &vision_msm, &inertial_msm);
    
    ba.set_initial_damping(0.);
    ba.prepare_to_optimize();
    
    ba.compute_inertial_terms();
    ba.compute_vision_terms();
    
    ba.initialize_normal_equations();
    ba.add_inertial_normal_equations();
    ba.add_vision_normal_equations();
    
    double vision_cost = ba.compute_vision_cost(bundle);
    double inertial_cost = ba.compute_inertial_cost(bundle);
    
#   include "data/JointNormalEquations.inc"

    EXPECT_NEAR_ARRAYS(ba.m_normal_eqns_lhs.data(), H_expected, 3*3*15*15, 1e-5);
    EXPECT_NEAR_ARRAYS(ba.m_normal_eqns_rhs.data(), g_expected, 3*15, 1e-5);
    
    EXPECT_NEAR(vision_cost, expected_vision_cost, 1e-5);
    EXPECT_NEAR(inertial_cost, expected_inertial_cost, 1e-5);
}
```

#### Further resources

https://code.google.com/p/googletest/wiki/FAQ

http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
