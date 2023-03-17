ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"
val NewTypeVersion = "0.4.4"
val zioVersion = "1.0.9"


lazy val root = (project in file("."))
  .settings(
    name := "HyperAI",
    libraryDependencies ++= Seq(
      "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
      "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1",
      "mysql" % "mysql-connector-java" % "8.0.27",
      "dev.zio" %% "zio" % zioVersion,
      "dev.zio" %% "zio-streams" % zioVersion,
      "com.zendesk" % "mysql-binlog-connector-java" % "0.27.6",
      "io.estatico" %% "newtype" % NewTypeVersion,
      "com.typesafe" % "config" % "1.4.2",
      "org.tpolecat" %% "doobie-core" % "1.0.0-RC1",
      "mysql" % "mysql-connector-java" % "5.1.44",
      "org.tpolecat" %% "doobie-hikari" % "1.0.0-RC1", // HikariCP transactor.
      "org.tpolecat" %% "doobie-specs2" % "1.0.0-RC1" % "test", // Specs2 support for typechecking statements.
      "org.tpolecat" %% "doobie-scalatest" % "1.0.0-RC1" % "test", // ScalaTest support for typechecking statements.
      //      "org.apache.logging.log4j" % "log4j-api-scala_2.13" % "12.0",
      "org.apache.logging.log4j" % "log4j-core" % "2.20.0",
      "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
      "ch.qos.logback" % "logback-classic" % "1.4.5"
    )
  )
