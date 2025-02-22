#include "frontend/PixelSelector2.h"
#include "internal/FrameHessian.h"
#include "internal/GlobalCalib.h"

using namespace ldso::internal;

namespace ldso {
    /**
     * @brief Construct a new Pixel Selector:: Pixel Selector object
     * Initialize PixelSelector with allocation of some space for image gradiant
     * @param w 
     * @param h 
     */
    PixelSelector::PixelSelector(int w, int h) {
        randomPattern = new unsigned char[w * h];
        std::srand(3141592);    // want to be deterministic.
        for (int i = 0; i < w * h; i++)
            randomPattern[i] = rand() & 0xFF; // 197, deterministic -> why it is random pattern? and why it uses bit operation?
        currentPotential = 3;
        gradHist = new int[100 * (1 + w / 32) * (1 + h / 32)]; // (w/32) + 1: number of width/32, error = 1, each grid can has 100 values -> don't have to be like this, I think..
        ths = new float[(w / 32) * (h / 32) + 100]; // I don't know the exact meaning of 100, I will find it later
        thsSmoothed = new float[(w / 32) * (h / 32) + 100];
    }
    /**
     * @brief Destroy the Pixel Selector:: Pixel Selector object
     * Destruct PixelSelector by deleting all of the dynamic arrays
     */
    PixelSelector::~PixelSelector() {
        delete[] randomPattern;
        delete[] gradHist;
        delete[] ths;
        delete[] thsSmoothed;
    }
    /**
     * @brief compute Gradient Histogram's threshold by pre-defined threshold(pixel count)
     * 
     * @param hist 
     * @param below 
     * @return int 
     */
    int computeHistQuantil(int *hist, float below) {
        int th = hist[0] * below + 0.5f; // get the threshold count by total count(hist[0])
        for (int i = 0; i < 90; i++) {
            th -= hist[i + 1]; // find the threshold bin by substraction
            if (th < 0) return i; // if find, return it
        }
        return 90; // bin is 0 ~ 49, so return 90 means not find
    }
    /**
     * @brief Make gradient histogram of the frame
     * 
     * @param fh 
     */
    void PixelSelector::makeHists(shared_ptr<FrameHessian> fh) {
        gradHistFrame = fh;
        float *mapmax0 = fh->absSquaredGrad[0]; // gradient map of original image

        int w = wG[0]; // in this function, only uses size of original image
        int h = hG[0];

        int w32 = w / 32; // num of width after divided into 32x32 grid
        int h32 = h / 32; // num of height after divided into 32x32 grid
        thsStep = w32; // threshold step(numnber of pixel) of one grid

        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float *map0 = mapmax0 + 32 * x + 32 * y * w; // map0 is the pointer for the current pixel of gradient map
                int *hist0 = gradHist;// + 50*(x+y*w32);, gradient histogram memory -> former code was keeping the full histogram map for all grids, but now don't have to be like this
                memset(hist0, 0, sizeof(int) * 50); // initialize the hist0, histogram range 0 ~ 49

                for (int j = 0; j < 32; j++)
                    for (int i = 0; i < 32; i++) {
                        int it = i + 32 * x;
                        int jt = j + 32 * y;
                        if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1) continue; // skip the edge of the image;in the makeImage function, already skipped edge regions
                        int g = sqrtf(map0[i + j * w]); // get the absSquaredGrad (dx^2, + dy^2), and square it to get gradient
                        if (g > 48) g = 48; // if the gradient is bigger than 48, cut off into 48
                        hist0[g + 1]++; // increase the count of gradient in the range of 1 ~ 49 -> b/c 0 is for the total count
                        hist0[0]++; // I don't know why it increases gradient 1 also, but in my opinion, it is for the self-gradient? NO, hist0[0] is the total count of image
                    }

                ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd; // calculate threshold for each grid
            }
        // calculate theSmoothed
        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float sum = 0, num = 0;
                if (x > 0) {
                    if (y > 0) {
                        num++;
                        sum += ths[x - 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x - 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x - 1 + (y) * w32];
                }

                if (x < w32 - 1) {
                    if (y > 0) {
                        num++;
                        sum += ths[x + 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x + 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x + 1 + (y) * w32];
                }

                if (y > 0) {
                    num++;
                    sum += ths[x + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x + (y + 1) * w32];
                }
                num++;
                sum += ths[x + y * w32];

                thsSmoothed[x + y * w32] = (sum / num) * (sum / num); // bind 3x3 threshold and get average theSmoothed for smoothe gradient
            }
    }
    /**
     * @brief make maps via gradient histogram and dynamic grid
     * 
     * @param fh 
     * @param map_out 
     * @param density 
     * @param recursionsLeft 
     * @param plot 
     * @param thFactor 
     * @return int 
     */
    int PixelSelector::makeMaps(const shared_ptr<FrameHessian> fh, float *map_out, float density,
                                int recursionsLeft, bool plot, float thFactor) {

        float numHave = 0;
        float numWant = density; // lvl0 --> 0.03 of total pixels
        float quotia;
        int idealPotential = currentPotential;

        if (fh != gradHistFrame) makeHists(fh); // make Gradient Histogram

        // select!
        Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor); // select pixels via Dynamic Grid, currentPotential = 3

        // sub-select!
        numHave = n[0] + n[1] + n[2]; // get the total num of selected pixels via adding each nums of image pyramids
        quotia = numWant / numHave; // calculate the ratio of target/selected pixels

        // by default we want to over-sample by 40% just to be sure.
        float K = numHave * (currentPotential + 1) * (currentPotential + 1);
        idealPotential = sqrtf(K / numWant) - 1;    // round down.
        if (idealPotential < 1) idealPotential = 1;
        // generate pixel selection map one more time after adjusting grid size
        if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1) {
            // re-sample to get more points!
            // potential needs to be smaller
            if (idealPotential >= currentPotential)
                idealPotential = currentPotential - 1;

            currentPotential = idealPotential;
            return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
        } else if (recursionsLeft > 0 && quotia < 0.25) {
            // re-sample to get less points!
            if (idealPotential <= currentPotential)
                idealPotential = currentPotential + 1;
            currentPotential = idealPotential;
            return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
        }

        int numHaveSub = numHave;
        if (quotia < 0.95) {
            int wh = wG[0] * hG[0];
            int rn = 0;
            unsigned char charTH = 255 * quotia;
            for (int i = 0; i < wh; i++) {
                if (map_out[i] != 0) {
                    if (randomPattern[rn] > charTH) {
                        map_out[i] = 0;
                        numHaveSub--;
                    }
                    rn++;
                }
            }
        }

        currentPotential = idealPotential;

        return numHaveSub;
    }
    /**
     * @brief Generating Pixel map via Dynamic Grid method
     * 
     * @param fh 
     * @param map_out 
     * @param pot 
     * @param thFactor 
     * @return Eigen::Vector3i 
     */
    Eigen::Vector3i PixelSelector::select(const shared_ptr<FrameHessian> fh, float *map_out, int pot,
                                          float thFactor) {
        Eigen::Vector3f const *const map0 = fh->dI; // dI[0] = pixel intensity;dI[1] = dx;dI[2] = dy
        // absSquaredGrad contains dx^2 + dy^2
        float *mapmax0 = fh->absSquaredGrad[0]; // pyramid level 0 (original)
        float *mapmax1 = fh->absSquaredGrad[1]; // pyramid level 1 (1/4)
        float *mapmax2 = fh->absSquaredGrad[2]; // pyramid level 2 (1/16)


        int w = wG[0]; // width of original image
        int w1 = wG[1]; //width of pyramid level 1 (1/4)
        int w2 = wG[2]; //width of pyramid level 2 (1/16)
        int h = hG[0]; // height of original image

        // random directions to prevent pixels to be crowded in certain regions
        const Vec2f directions[16] = {
                Vec2f(0, 1.0000),
                Vec2f(0.3827, 0.9239),
                Vec2f(0.1951, 0.9808),
                Vec2f(0.9239, 0.3827),
                Vec2f(0.7071, 0.7071),
                Vec2f(0.3827, -0.9239),
                Vec2f(0.8315, 0.5556),
                Vec2f(0.8315, -0.5556),
                Vec2f(0.5556, -0.8315),
                Vec2f(0.9808, 0.1951),
                Vec2f(0.9239, -0.3827),
                Vec2f(0.7071, -0.7071),
                Vec2f(0.5556, 0.8315),
                Vec2f(0.9808, -0.1951),
                Vec2f(1.0000, 0.0000),
                Vec2f(0.1951, -0.9808)};

        memset(map_out, 0, w * h * sizeof(PixelSelectorStatus)); // initialize map_out that can store pyramid levels of selected pixels

        float dw1 = setting_gradDownweightPerLevel; // 0.75
        float dw2 = dw1 * dw1; // multiply down weight (0.75) as pyramid levels go high -> intend to select many pixels from the lowest level, the original image

        int n3 = 0, n2 = 0, n4 = 0; // num of pixel selected from each levels
        for (int y4 = 0; y4 < h; y4 += (4 * pot)) // loop for pyramid level 2 (1/16)
            for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
                int my3 = std::min((4 * pot), h - y4);
                int mx3 = std::min((4 * pot), w - x4);
                int bestIdx4 = -1;
                float bestVal4 = 0;
                Vec2f dir4 = directions[randomPattern[n2] & 0xF];
                for (int y3 = 0; y3 < my3; y3 += (2 * pot)) // loop for pyramid level 1 (1/4)
                    for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
                        int x34 = x3 + x4;
                        int y34 = y3 + y4;
                        int my2 = std::min((2 * pot), h - y34);
                        int mx2 = std::min((2 * pot), w - x34);
                        int bestIdx3 = -1;
                        float bestVal3 = 0;
                        Vec2f dir3 = directions[randomPattern[n2] & 0xF];
                        for (int y2 = 0; y2 < my2; y2 += pot) // loop for pyramid level 0 (original)
                            for (int x2 = 0; x2 < mx2; x2 += pot) {
                                int x234 = x2 + x34;
                                int y234 = y2 + y34;
                                int my1 = std::min(pot, h - y234);
                                int mx1 = std::min(pot, w - x234);
                                int bestIdx2 = -1;
                                float bestVal2 = 0;
                                Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                                for (int y1 = 0; y1 < my1; y1 += 1) // select pixel from current grid
                                    for (int x1 = 0; x1 < mx1; x1 += 1) {
                                        assert(x1 + x234 < w);
                                        assert(y1 + y234 < h);
                                        int idx = x1 + x234 + w * (y1 + y234); // index of current pixel
                                        int xf = x1 + x234; // current x
                                        int yf = y1 + y234; // current y

                                        if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4) continue; //skip the edges


                                        float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep]; // why bit operation?
                                        float pixelTH1 = pixelTH0 * dw1; // multiply down weight to reduce the probability
                                        float pixelTH2 = pixelTH1 * dw2; // multiply down weight to reduce the probability


                                        float ag0 = mapmax0[idx]; // ag0 -> current pixel's absolute squared gradient (dx^2 + dy^2)
                                        if (ag0 > pixelTH0 * thFactor) { // if current pixel's absolute squared gradient is bigger than the threshold -> check original
                                            Vec2f ag0d = map0[idx].tail<2>(); // ag0d -> current pixel's dx, dy
                                            float dirNorm = fabsf((float) (ag0d.dot(dir2))); // dirNorm : random direction * (dx, dy)
                                            if (!setting_selectDirectionDistribution) dirNorm = ag0; // parameter is set true, so not enter this code

                                            if (dirNorm > bestVal2) { // if a pixel that satisfies the condition in the original image's grid, add it and escape the loop
                                                bestVal2 = dirNorm;
                                                bestIdx2 = idx;
                                                bestIdx3 = -2; // -2 means non-activated
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx3 == -2) continue; // if a pixel is selected at the original image, move to the next grid

                                        float ag1 = mapmax1[(int) (xf * 0.5f + 0.25f) + (int) (yf * 0.5f + 0.25f) * w1]; // (1/4) image's absolute squared gradient
                                        if (ag1 > pixelTH1 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir3)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag1;

                                            if (dirNorm > bestVal3) { // if a pixel that satisfies the condition in the (1/4) image's grid, add it and escape the loop
                                                bestVal3 = dirNorm;
                                                bestIdx3 = idx;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx4 == -2) continue; // if a pixel is selected at the (1/4) image, move to the next grid

                                        float ag2 = mapmax2[(int) (xf * 0.25f + 0.125) +
                                                            (int) (yf * 0.25f + 0.125) * w2]; // (1/16) image's absolute squared gradient
                                        if (ag2 > pixelTH2 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir4)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag2;

                                            if (dirNorm > bestVal4) { // if a pixel that satisfies the condition in the (1/16) image's grid, add it and escape the loop
                                                bestVal4 = dirNorm;
                                                bestIdx4 = idx;
                                            }
                                        }
                                    }

                                if (bestIdx2 > 0) { // if a pixel is selected from the original image
                                    map_out[bestIdx2] = 1; // replace 0 with 1 if a pixel is selected from the original image
                                    bestVal3 = 1e10;
                                    n2++; // num of original image's selected pixels
                                }
                            }

                        if (bestIdx3 > 0) { // if a pixel is selected from the (1/4) image
                            map_out[bestIdx3] = 2; // replace 0 with 2 if a pixel is selected from the (1/4) image
                            bestVal4 = 1e10;
                            n3++; // num of (1/4) image's selected pixels
                        }
                    }

                if (bestIdx4 > 0) { // if a pixel is selected from the (1/16) image
                    map_out[bestIdx4] = 4; // replace 0 with 4 if a pixel is selected from the (1/16) image
                    n4++; // num of (1/16) image's selected pixels
                }
            }


        return Eigen::Vector3i(n2, n3, n4); // return each nums of selected pixels from each pyramids => we can get total num of selected pixels by sum of them
    }

}