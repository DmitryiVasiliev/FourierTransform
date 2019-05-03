package sample;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.opencv.imgproc.Imgproc;

import javax.security.sasl.SaslServer;

/**
 * The controller associated to the only view of our application. The
 * application logic is implemented here. It handles the button for opening an
 * image and perform all the operation related to the Fourier transformation and
 * antitransformation.
 *
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 2.0 (2017-03-10)
 * @since 1.0 (2013-12-11)
 */
public class Controller {
    // images to show in the view
    @FXML
    private ImageView originalImage;
    @FXML
    private ImageView transformedImage;
    @FXML
    private ImageView antitransformedImage;
    // a FXML button for performing the transformation
    @FXML
    private Button transformButton;
    // a FXML button for performing the antitransformation
    @FXML
    private Button antitransformButton;

    // the main stage
    private Stage stage;
    // the JavaFX file chooser
    private FileChooser fileChooser;
    // support variables
    private Mat image;
    private List<Mat> planes;
    // the final complex image
    private Mat complexImage;
    int xMin, xMax, yMin, yMax;

    private int count = 2;


    ArrayList<Mat> mats = new ArrayList<>();

    private Mat image2;

    /**
     * Init the needed variables
     */
    protected void init() {
        this.fileChooser = new FileChooser();
        this.image = new Mat();
        this.planes = new ArrayList<>();
        this.complexImage = new Mat();
    }

    /**
     * Load an image from disk
     */
    @FXML
    protected void loadImage() {

        Mat image2 = new Mat(), image3 = new Mat(), image1 = new Mat();
        // show the open dialog window
        File file = this.fileChooser.showOpenDialog(this.stage);
        if (file != null) {
            // read the image in gray scale
            this.image = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            image.copyTo(image1);
            mats.add(image1);
            xMin = this.image.cols() / 2 - 5;
            yMin = this.image.rows() / 2 - 5;
            xMax = this.image.cols() / 2 + 5;
            yMax = this.image.rows() / 2 + 5;
            // make black hole
            Imgproc.rectangle(this.image, new Point((xMin), (yMin)), new Point(xMax, (yMax)), new Scalar(0, 0, 0, 0), -1);
            this.image.copyTo(image2);
            mats.add(image2);
            double aver = 0;
            for (int i = 0; i < this.image.rows(); i++) {
                for (int j = 0; j < this.image.cols(); j++)
                    if (i >= yMin && i <= yMax && j >= xMin && j <= xMax) {

                    } else {
                        aver += this.image.get(i, j)[0];
                    }
            }
            // fill hole average
            for (int i = 0; i < this.image.rows(); i++) {
                for (int j = 0; j < this.image.cols(); j++)
                    if (i >= yMin && i <= yMax && j >= xMin && j <= xMax) {
                        this.image.put(i, j, (aver / (this.image.cols() * this.image.rows() - 9)));
                    }
            }
            this.image.copyTo(image3);
            mats.add(image3);

            // show the image
            this.updateImageView(originalImage, Utils.mat2Image(image3));
            // set a fixed width
            this.originalImage.setFitWidth(250);
            // preserve image ratio
            this.originalImage.setPreserveRatio(true);
            // update the UI
            this.transformButton.setDisable(false);

            // empty the image planes and the image views if it is not the first
            // loaded image
            if (!this.planes.isEmpty()) {
                this.planes.clear();
                this.transformedImage.setImage(null);
                this.antitransformedImage.setImage(null);
            }

        }
    }

    /**
     * The action triggered by pushing the button for apply the dft to the
     * loaded image
     */
    @FXML
    protected void transformImage() {
        this.planes.clear();
        // optimize the dimension of the loaded image
        Mat padded = this.optimizeImageDim(this.image);
        padded.convertTo(padded, CvType.CV_32F);
        // prepare the image planes to obtain the complex image
        this.planes.add(padded);
        this.planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        // prepare a complex image for performing the dft
        Core.merge(this.planes, this.complexImage);

        // dft
        Core.dft(this.complexImage, this.complexImage);
        // optimize the image resulting from the dft operation
        Mat magnitude = this.createOptimizedMagnitude(this.complexImage);
        //    System.out.println(magnitude.dump());
        //   System.out.println("-----------------------------------------------------------------");

        // show the result of the transformation as an image
        this.updateImageView(transformedImage, Utils.mat2Image(magnitude));
        // set a fixed width
        this.transformedImage.setFitWidth(250);
        // preserve image ratio
        this.transformedImage.setPreserveRatio(true);

        // enable the button for performing the antitransformation
        this.antitransformButton.setDisable(false);
        // disable the button for applying the dft
        this.transformButton.setDisable(false);
    }

    private void fourierTransformfromNComponent(Mat mat) {
        for (int i = 0; i < mat.cols(); i++) {
            mat.put(0, count, 0, 0);
        }
        for (int i = 1; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                mat.put(i, j, 0, 0);
            }

        }
    }

    /**
     * The action triggered by pushing the button for apply the inverse dft to
     * the loaded image
     */
    @FXML
    protected void antitransformImage() {
        fourierTransformfromNComponent(this.complexImage);
        Core.idft(this.complexImage, this.complexImage);

        Mat restoredImage = new Mat();
        Core.split(this.complexImage, this.planes);
        Core.normalize(this.planes.get(0), restoredImage, 0, 255, Core.NORM_MINMAX);

        // move back the Mat to 8 bit, in order to proper show the result
        restoredImage.convertTo(restoredImage, CvType.CV_8U);
        //fill hole from image with N spectrum component
        for (int i = yMin; i < yMax; i++) {
            for (int j = xMin; j < xMax; j++) {
                this.image.put(i, j, restoredImage.get(i, j));
            }
        }
        this.updateImageView(originalImage, Utils.mat2Image(this.image));

        this.updateImageView(antitransformedImage, Utils.mat2Image(restoredImage));
        // set a fixed width
        this.antitransformedImage.setFitWidth(250);
        // preserve image ratio
        this.antitransformedImage.setPreserveRatio(true);

        // disable the button for performing the antitransformation
        this.antitransformButton.setDisable(false);
        this.planes.clear();
        count++;
    }

    /**
     * Optimize the image dimensions
     *
     * @param image the {@link Mat} to optimize
     * @return the image whose dimensions have been optimized
     */
    private Mat optimizeImageDim(Mat image) {
        // init
        Mat padded = new Mat();
        // get the optimal rows size for dft
        int addPixelRows = Core.getOptimalDFTSize(image.rows());
        // get the optimal cols size for dft
        int addPixelCols = Core.getOptimalDFTSize(image.cols());
        // apply the optimal cols and rows size to the image
        Core.copyMakeBorder(image, padded, 0, addPixelRows - image.rows(), 0, addPixelCols - image.cols(),
                Core.BORDER_CONSTANT, Scalar.all(0));

        return padded;
    }

    /**
     * Optimize the magnitude of the complex image obtained from the DFT, to
     * improve its visualization
     *
     * @param complexImage the complex image obtained from the DFT
     * @return the optimized image
     */
    private Mat createOptimizedMagnitude(Mat complexImage) {
        // init
        List<Mat> newPlanes = new ArrayList<>();
        Mat mag = new Mat();
        // split the comples image in two planes
        Core.split(complexImage, newPlanes);
        // compute the magnitude
        Core.magnitude(newPlanes.get(0), newPlanes.get(1), mag);

        // move to a logarithmic scale
        Core.add(Mat.ones(mag.size(), CvType.CV_32F), mag, mag);
        Core.log(mag, mag);
        // optionally reorder the 4 quadrants of the magnitude image
        //  this.shiftDFT(mag);
        // normalize the magnitude image for the visualization since both JavaFX
        // and OpenCV need images with value between 0 and 255
        // convert back to CV_8UC1
        mag.convertTo(mag, CvType.CV_8UC1);
        Core.normalize(mag, mag, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
        // you can also write on disk the resulting image...
        // Imgcodecs.imwrite("../magnitude.png", mag);

        return mag;
    }

    /**
     * Reorder the 4 quadrants of the image representing the magnitude, after
     * the DFT
     *
     * @param image the {@link Mat} object whose quadrants are to reorder
     */
    private void shiftDFT(Mat image) {
        image = image.submat(new Rect(0, 0, image.cols() & -2, image.rows() & -2));
        int cx = image.cols() / 2;
        int cy = image.rows() / 2;

        Mat q0 = new Mat(image, new Rect(0, 0, cx, cy));
        Mat q1 = new Mat(image, new Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(image, new Rect(0, cy, cx, cy));
        Mat q3 = new Mat(image, new Rect(cx, cy, cx, cy));

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

    /**
     * Set the current stage (needed for the FileChooser modal window)
     *
     * @param stage the stage
     */
    public void setStage(Stage stage) {
        this.stage = stage;
    }


    /**
     * Update the {@link ImageView} in the JavaFX main thread
     *
     * @param view  the {@link ImageView} to update
     * @param image the {@link Image} to show
     */
    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

}


//package sample;
//
//import javafx.event.ActionEvent;
//import javafx.fxml.FXML;
//import javafx.fxml.FXMLLoader;
//import javafx.scene.Parent;
//import javafx.scene.Scene;
//import javafx.scene.control.Button;
//import javafx.scene.control.Label;
//import javafx.scene.image.Image;
//import javafx.scene.image.ImageView;
//import javafx.stage.FileChooser;
//import javafx.stage.Stage;
//import org.opencv.core.*;
//import org.opencv.imgcodecs.Imgcodecs;
//import org.opencv.imgproc.Imgproc;
//
//import java.io.File;
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.List;
//
//public class Controller {
//    @FXML
//    private ImageView originalImage;
//    @FXML
//    private ImageView transformedImage;
//    @FXML
//    private ImageView antitransformedImage;
//    // a FXML button for performing the transformation
//    @FXML
//    private Button transformButton;
//    // a FXML button for performing the antitransformation
//    @FXML
//    private Button antitransformButton;
//
//    // the main stage
//    private Stage stage;
//    // the JavaFX file chooser
//    private FileChooser fileChooser;
//    // support variables
//    private Mat image;
//    private List<Mat> planes;
//    // the final complex image
//    private Mat complexImage;
//
//    private Mat imR;
//    private Mat imG;
//    private Mat imB;
//    private boolean checkChannels;
//int i = 1;
//
//    /**
//     * Init the needed variables
//     */
//    protected void init() {
//        this.fileChooser = new FileChooser();
//        this.image = new Mat();
//        this.imR = new Mat();
//        this.imG = new Mat();
//        this.imB = new Mat();
//        this.checkChannels = false;
//        this.planes = new ArrayList<>();
//        this.complexImage = new Mat();
//    }
//
//    /**
//     * Load an image from disk
//     */
//    @FXML
//    protected void loadImage() throws InterruptedException {
//        // show the open dialog window
//
//        File file = this.fileChooser.showOpenDialog(this.stage);
//        if (file != null) {
//            // read the image in gray scale
//            double r = 0;
//            double g = 0;
//            double b = 0;
//           // this.image = Imgcodecs.imread(file.getAbsolutePath());
//
//           // Imgproc.rectangle(this.image, new Point((this.image.cols() / 2 - 10), (this.image.rows() / 2 - 2)), new Point((this.image.cols() / 2) + 20, (this.image.rows() / 2 + 2)), new Scalar(0, 0, 0, 0), -1);
////            Imgproc.rectangle(this.image, new Point((this.image.cols() / 2 - 1), (this.image.rows() / 2 - 1)), new Point((this.image.cols() / 2) + 1, (this.image.rows() / 2 + 1)), new Scalar(0, 0, 0, 0), -1);
//            this.checkChannels = image.channels() == 1 ? true : false;
////            for (int i = 0; i < 150; i++) {
////                for (int j = 0; j < 150; j++) {
////                    if (this.image.get(i, j)[0] != 0 && this.image.get(i, j)[1] != 0 && this.image.get(i, j)[2] != 0) {
////                        r += this.image.get(i, j)[0];
////                        g += this.image.get(i, j)[1];
////                        b += this.image.get(i, j)[2];
////                    }
////                }
////            }
////            r = r / (150 * 150);
////            g = g / (150 * 150);
////            b = b / (150 * 150);
////            for (int i = 73; i < 78; i++) {
////                for (int j = 65; j < 96; j++) {
////
////                    image.put(i, j, r, g, b);
////
////                }
////            }
//              this.image = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
//            for(int i =0; i < 10; i++){
//                for(int j=0;j<10;j++){
//                    System.out.print(this.image.get(i,j)+" | ");
//                }
//                System.out.println("");
//            }
//            Imgproc.rectangle(this.image, new Point((this.image.cols() / 2 - 1), (this.image.rows() / 2 - 1)), new Point((this.image.cols() / 2) + 1, (this.image.rows() / 2 + 1)), new Scalar(0, 0, 0, 0), -1);
//            for(int i =0; i < 10; i++){
//                for(int j=0;j<10;j++){
//                    System.out.print(this.image.get(i,j)+" | ");
//                }
//                System.out.println("");
//            }
//            // show the image
//            this.updateImageView(originalImage, Utils.mat2Image(this.image));
//            // set a fixed width
//            this.originalImage.setFitWidth(250);
//            // preserve image ratio
//            this.originalImage.setPreserveRatio(true);
//            // update the UI
//            this.transformButton.setDisable(false);
//
//            // empty the image planes and the image views if it is not the first
//            // loaded image
//            if (!this.planes.isEmpty()) {
//                this.planes.clear();
//                this.transformedImage.setImage(null);
//                this.antitransformedImage.setImage(null);
//            }
//
//        }
//    }
//
//    /**
//     * The action triggered by pushing the button for apply the dft to the
//     * loaded image
//     */
//    @FXML
//    protected void transformImage() throws InterruptedException, IOException {
//       i++;
//        dtfForColorImage(i);
//        Mat mat = antitransformColorImage();
//        setPixels(mat);
//        this.updateImageView(originalImage,Utils.mat2Image(this.image));
//        this.updateImageView(antitransformedImage,Utils.mat2Image(mat));
//
//    }
//
//    private void testForen(int t) throws InterruptedException, IOException {
//
//        for (int i = 1; i < t; i++) {
//            dtfForColorImage(i);
//            Mat mat = antitransformColorImage();
//           setPixels(mat);
//           this.updateImageView(this.antitransformedImage,Utils.mat2Image(this.image));
//          // Thread.sleep(5000);
//          // showForm(this.image,i);
//        }
//
//    }
//
//
//    private void setPixels(Mat mat){
//        for (int i = 73; i < 78; i++) {
//            for (int j = 65; j < 96; j++) {
//                this.image.put(i,j,mat.get(i,j));
//
//            }
//        }
//
//    }
//
//    private Mat dtfForColorImage(int k) throws InterruptedException {
//        int i = 0;
//        for(int t =0; t<150;t++){
//            for(int o =0; o<150;o++){
//                imR.put(t,o,0,0);
//                imG.put(t,o,0,0);
//                imB.put(t,o,0,0);
//            }
//        }
//        // optimize the dimension of the loaded image
//        Mat padded = this.optimizeImageDim(this.image);
//        ArrayList<Mat> mats = new ArrayList<>();
//        Core.split(padded, mats);
//
//        for (Mat mat : mats) {
//            if (!this.planes.isEmpty())
//                this.planes.clear();
//            mat.convertTo(mat, CvType.CV_64FC1);
//            this.planes.add(mat);
//            this.planes.add(Mat.zeros(mat.size(), CvType.CV_64FC1));
//            if (i == 0) {
//                Core.merge(this.planes, imR);
//                Core.dft(imR, imR);
//            }
//            if (i == 1) {
//                Core.merge(this.planes, imG);
//                Core.dft(imG, imG);
//            }
//            if (i == 2) {
//                Core.merge(this.planes, imB);
//                Core.dft(imB, imB);
//            }
//
//            i++;
//
//        }
//
//        ArrayList<Mat> mats1 = new ArrayList<>();
//        Mat test = new Mat();
//        test.convertTo(test, CvType.CV_64FC1);
//        mats1.add(imR);
//        mats1.add(imG);
//        mats1.add(imB);
//        Core.merge(mats1, test);
//        ArrayList<Mat> magn = new ArrayList<>();
//        // optimize the image resulting from the dft operation
//        for (Mat mat : mats1) {
//            Mat magnitude = this.createOptimizedMagnitude(mat);
//            magn.add(magnitude);
//        }
//
//        Mat magnitude = new Mat();
//        magnitude.convertTo(magnitude, CvType.CV_64FC1);
//        Core.merge(magn, magnitude);
//
//
//
//
//        for (int y = k; y <150; y++) {
//            for (int g = 0; g < 150; g++) {
//                imR.put(y, g, 0, 0);
//                imG.put(y, g, 0, 0);
//                imB.put(y, g, 0, 0);
//            }
//        }
//
//
//        // show the result of the transformation as an image
//        this.updateImageView(transformedImage, Utils.mat2Image(magnitude));
//        // set a fixed width
//        this.transformedImage.setFitWidth(250);
//        // preserve image ratio
//        this.transformedImage.setPreserveRatio(true);
//
//        // enable the button for performing the antitransformation
//      //  this.antitransformButton.setDisable(false);
//        // disable the button for applying the dft
//    //    this.transformButton.setDisable(true);
//        return magnitude;
//    }
//
//    private void dtfForGrayImage() {
//        Mat padded = this.optimizeImageDim(this.image);
//        padded.convertTo(padded, CvType.CV_64FC1);
//        // prepare the image planes to obtain the complex image
//        this.planes.add(padded);
//        this.planes.add(Mat.zeros(padded.size(), CvType.CV_64FC1));
//        // prepare a complex image for performing the dft
//        Core.merge(this.planes, this.complexImage);
//
//        //    dft
//        Core.dft(this.complexImage, this.complexImage);
//        Mat magnitude = this.createOptimizedMagnitude(this.complexImage);
//
//        // show the result of the transformation as an image
//        this.updateImageView(transformedImage, Utils.mat2Image(magnitude));
//        // set a fixed width
//        this.transformedImage.setFitWidth(250);
//        // preserve image ratio
//        this.transformedImage.setPreserveRatio(true);
//
//        // enable the button for performing the antitransformation
//        this.antitransformButton.setDisable(false);
//        // disable the button for applying the dft
//        this.transformButton.setDisable(true);
//    }
//
//    /**
//     * The action triggered by pushing the button for apply the inverse dft to
//     * the loaded image
//     */
//    @FXML
//    protected void antitransformImage() {
//        if (checkChannels)
//            antitransformGrayImage();
//        else
//            antitransformColorImage();
//    }
//
//    private Mat antitransformColorImage() {
//
//        Core.idft(this.imR, this.imR);
//        Core.idft(this.imG, this.imG);
//        Core.idft(this.imB, this.imB);
//
//        Mat restoredImageR = new Mat();
//        Mat restoredImageG = new Mat();
//        Mat restoredImageB = new Mat();
//        Core.split(this.imR, this.planes);
//        Core.normalize(this.planes.get(0), restoredImageR, 0, 255, Core.NORM_MINMAX);
//        this.planes.clear();
//        Core.split(this.imG, this.planes);
//        Core.normalize(this.planes.get(0), restoredImageG, 0, 255, Core.NORM_MINMAX);
//        this.planes.clear();
//        Core.split(this.imB, this.planes);
//        Core.normalize(this.planes.get(0), restoredImageB, 0, 255, Core.NORM_MINMAX);
//
//        // move back the Mat to 8 bit, in order to proper show the result
//        restoredImageR.convertTo(restoredImageR, CvType.CV_8U);
//        restoredImageG.convertTo(restoredImageG, CvType.CV_8U);
//        restoredImageB.convertTo(restoredImageB, CvType.CV_8U);
//        ArrayList<Mat> mats = new ArrayList<>();
//        mats.add(restoredImageR);
//        mats.add(restoredImageG);
//        mats.add(restoredImageB);
//        Mat restoredImage = new Mat();
//        Core.merge(mats, restoredImage);
//
//  //        this.updateImageView(antitransformedImage, Utils.mat2Image(restoredImage));
//        // set a fixed width
//        this.antitransformedImage.setFitWidth(250);
//        // preserve image ratio
//        this.antitransformedImage.setPreserveRatio(true);
//
//        // disable the button for performing the antitransformation
//        this.antitransformButton.setDisable(true);
//        return restoredImage;
//    }
//
//    private void antitransformGrayImage() {
//        Core.idft(this.complexImage, this.complexImage);
//
//        Mat restoredImage = new Mat();
//        Core.split(this.complexImage, this.planes);
//        Core.normalize(this.planes.get(0), restoredImage, 0, 255, Core.NORM_MINMAX);
//
//        // move back the Mat to 8 bit, in order to proper show the result
//        restoredImage.convertTo(restoredImage, CvType.CV_8U);
//
//        this.updateImageView(antitransformedImage, Utils.mat2Image(restoredImage));
//        // set a fixed width
//        this.antitransformedImage.setFitWidth(250);
//        // preserve image ratio
//        this.antitransformedImage.setPreserveRatio(true);
//
//        // disable the button for performing the antitransformation
//        this.antitransformButton.setDisable(true);
//    }
//
//    /**
//     * Optimize the image dimensions
//     *
//     * @param image the {@link Mat} to optimize
//     * @return the image whose dimensions have been optimized
//     */
//    private Mat optimizeImageDim(Mat image) {
//        // init
//        Mat padded = new Mat();
//        // get the optimal rows size for dft
//        int addPixelRows = Core.getOptimalDFTSize(image.rows());
//        // get the optimal cols size for dft
//        int addPixelCols = Core.getOptimalDFTSize(image.cols());
//        // apply the optimal cols and rows size to the image
//        Core.copyMakeBorder(image, padded, 0, addPixelRows - image.rows(), 0, addPixelCols - image.cols(),
//                Core.BORDER_CONSTANT, Scalar.all(0));
//
//        return padded;
//    }
//
//    /**
//     * Optimize the magnitude of the complex image obtained from the DFT, to
//     * improve its visualization
//     *
//     * @param complexImage the complex image obtained from the DFT
//     * @return the optimized image
//     */
//    private Mat createOptimizedMagnitude(Mat complexImage) {
//        // init
//        List<Mat> newPlanes = new ArrayList<>();
//        Mat mag = new Mat();
//        // split the comples image in two planes
//        Core.split(complexImage, newPlanes);
//        // compute the magnitude
//        Core.magnitude(newPlanes.get(0), newPlanes.get(1), mag);
//
//        // move to a logarithmic scale
//        Core.add(Mat.ones(mag.size(), CvType.CV_64FC1), mag, mag);
//        Core.log(mag, mag);
//        // optionally reorder the 4 quadrants of the magnitude image
//        this.shiftDFT(mag);
//        // normalize the magnitude image for the visualization since both JavaFX
//        // and OpenCV need images with value between 0 and 255
//        // convert back to CV_8UC1
//        mag.convertTo(mag, CvType.CV_8UC1);
//        Core.normalize(mag, mag, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
//
//        // you can also write on disk the resulting image...
//        // Imgcodecs.imwrite("../magnitude.png", mag);
//
//        return mag;
//    }
//
//    /**
//     * Reorder the 4 quadrants of the image representing the magnitude, after
//     * the DFT
//     *
//     * @param image the {@link Mat} object whose quadrants are to reorder
//     */
//    private void shiftDFT(Mat image) {
//        image = image.submat(new Rect(0, 0, image.cols() & -2, image.rows() & -2));
//        int cx = image.cols() / 2;
//        int cy = image.rows() / 2;
//
//        Mat q0 = new Mat(image, new Rect(0, 0, cx, cy));
//        Mat q1 = new Mat(image, new Rect(cx, 0, cx, cy));
//        Mat q2 = new Mat(image, new Rect(0, cy, cx, cy));
//        Mat q3 = new Mat(image, new Rect(cx, cy, cx, cy));
//
//        Mat tmp = new Mat();
//        q0.copyTo(tmp);
//        q3.copyTo(q0);
//        tmp.copyTo(q3);
//
//        q1.copyTo(tmp);
//        q2.copyTo(q1);
//        tmp.copyTo(q2);
//    }
//
//    /**
//     * Set the current stage (needed for the FileChooser modal window)
//     *
//     * @param stage the stage
//     */
//    public void setStage(Stage stage) {
//        this.stage = stage;
//    }
//
//
//    /**
//     * Update the {@link ImageView} in the JavaFX main thread
//     *
//     * @param view  the {@link ImageView} to update
//     * @param image the {@link Image} to show
//     */
//    private void updateImageView(ImageView view, Image image) {
//        Utils.onFXThread(view.imageProperty(), image);
//    }
//}
