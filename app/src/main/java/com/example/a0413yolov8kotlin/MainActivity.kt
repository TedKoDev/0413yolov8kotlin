package com.example.a0413yolov8kotlin

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.media.ExifInterface
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.*
import java.util.*

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var btnimage: Button
    private lateinit var takeBtn: Button
    private lateinit var textView: TextView

    private val dataProcess = DataProcess(context = this)
    private lateinit var session: OrtSession
    private lateinit var ortEnvironment: OrtEnvironment

    // Paint 객체 생성
    val paint = Paint()

    lateinit var bitmap: Bitmap

    companion object {
        private const val PICK_IMAGE_REQUEST = 1
        private const val ACTION_IMAGE_CAPTURE = 102
        private const val CAMERA_PERMISSION_REQUEST_CODE = 102
    }

    // 감지된 객체마다 다른 색상을 설정하기 위한 리스트 생성
    val colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK, Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.LTGRAY
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        btnimage = findViewById(R.id.btnimage)
        takeBtn = findViewById(R.id.takeBtn)
        textView = findViewById(R.id.textView)

        dataProcess.loadModel() // onnx 모델 불러오기
        dataProcess.loadLabel() // coco txt 파일 불러오기

        try {
            // ortEnvironment 변수 초기화
            ortEnvironment = OrtEnvironment.getEnvironment()
            session = ortEnvironment.createSession(
                this.filesDir.absolutePath.toString() + "/" + DataProcess.FILE_NAME,
                OrtSession.SessionOptions()
            )
        } catch (e: OrtException) {
            // 예외 처리: ONNX 모델 로드 및 세션 생성 시 예외 처리
            e.printStackTrace()
            // 예외 처리에 따른 처리 로직 추가
        }

        btnimage.setOnClickListener {
            openGallery()
        }

        takeBtn.setOnClickListener {
            takePhoto()
        }

        // 카메라 권한 요청
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
                // 사용자가 권한을 거부하고 "다시 묻지 않음" 옵션을 선택한 경우에 대한 처리 로직 추가
            } else {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
            }
        }
    }


    private fun takePhoto() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, ACTION_IMAGE_CAPTURE)
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, PICK_IMAGE_REQUEST)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == Activity.RESULT_OK && data != null) {
            // 받아온 이미지를 uri 변수에 저장
            val uri = data?.data

            // MediaStore를 사용하여 uri로부터 이미지를 bitmap 변수에 저장
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

            // 이미지의 방향 정보를 읽어옴
            val inputStream = contentResolver.openInputStream(uri!!)
            val exif = ExifInterface(inputStream!!)
            val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)

            //이미지 회전 방지코드
            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(270f)
            }

            //이미지 회전
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

            // 이미지 처리 함수 호출
            imageProcess(bitmap)

        } else if (requestCode == ACTION_IMAGE_CAPTURE) {
            // 사진을 찍은 결과값을 bitmap 변수에 저장
            bitmap = data?.extras?.get("data") as Bitmap

            //이미지 퀄리티 높이기
            val stream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)
            val byteArray = stream.toByteArray()
            bitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)

            //이미지의 방향 정보를 읽어옴
            val exif = ExifInterface(ByteArrayInputStream(byteArray))
            val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)

            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(270f)
            }
            //이미지 회전
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

            // 이미지 처리 함수 호출
            imageProcess(bitmap)
        }

    }


    private fun imageProcess(bitmap: Bitmap) {

        val bitmap2 = dataProcess.imageToBitmap(bitmap)
        val predictedClassNames = ArrayList<String>() // ArrayList to store predicted class names

        val floatBuffer = dataProcess.bitmapToFloatBuffer(bitmap2)
        val inputName = session.inputNames.iterator().next() // session 이름

        // 모델의 요구 입력값 [1 3 640 640] [배치 사이즈, 픽셀(RGB), 너비, 높이], 모델마다 크기는 다를 수 있음.
        val shape = longArrayOf(
            DataProcess.BATCH_SIZE.toLong(),
            DataProcess.PIXEL_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong()
        )
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
        val resultTensor = session.run(Collections.singletonMap(inputName, inputTensor))
        val outputs = resultTensor.get(0).value as Array<*> // [1 84 8400]
        val results = dataProcess.outputsToNPMSPredictions(outputs)

        val canvas = Canvas(bitmap2)

        dataProcess.loadLabel()

        // 사각형 그리기 x1, y1, x2, y2
        for (result in results) {

            val predictedClassIndex = result.classIndex

            // Accessing the properties from the model's output
            val x1 = result.rectF.left
            val y1 = result.rectF.top
            val x2 = result.rectF.right
            val y2 = result.rectF.bottom

            val rect = RectF(
                x1,
                y1,
                x2 ,
                y2
            )

            Log.d("Predicted Class", "x1 : $x1, y1 : $y1, x2 : $x2, y2 : $y2")

            // 사각형 그리기
            paint.color = colors[Random().nextInt(colors.size)]
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 5f
            canvas.drawRect(rect, paint)

            // 검출 된 객체의 이름과 퍼센트 표시
            paint.color = Color.WHITE
            paint.style = Paint.Style.FILL
            paint.textSize = 50f
            canvas.drawText( dataProcess.classes[predictedClassIndex] + " " + result.score, x1, y1+30f , paint)

            // Retrieving the corresponding class label from the `classes` array
            if (predictedClassIndex >= 0 && predictedClassIndex < dataProcess.classes.size) {
                val predictedClassName = dataProcess.classes[predictedClassIndex]
                Log.d("Predicted Class", predictedClassName)
                // Add predicted class name to the ArrayList
                predictedClassNames.add(predictedClassName)
            }
        }

        imageView.setImageBitmap(bitmap2)
        val predictedClassesText = predictedClassNames.toString()
        textView.text = predictedClassesText
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            // ortEnvironment 및 session 리소스 해제
            session.close() // session 리소스 해제
            ortEnvironment.close() // ortEnvironment 리소스 해제
        } catch (e: OrtException) {
            e.printStackTrace()
            // 예외 처리에 따른 처리 로직 추가
        }
    }
}
